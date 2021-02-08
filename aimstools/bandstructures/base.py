from aimstools.misc import *
from aimstools.postprocessing import FHIAimsOutputReader

from ase.dft.kpoints import parse_path_string, BandPath

from collections import namedtuple
import re

import numpy as np


class BandStructureBaseClass(FHIAimsOutputReader):
    def __init__(self, outputfile) -> None:
        super().__init__(outputfile)
        assert self.is_converged, "Calculation did not converge."
        tasks = {x for x in self.control["tasks"] if "band structure" in x}
        accepted_tasks = set(["band structure", "mulliken-projected band structure"])
        assert any(
            x in accepted_tasks for x in tasks
        ), "Band structure task not accepted."
        self.tasks = tasks
        self.task = None
        self._energy_reference = "not specified"
        self.band_sections = self.__set_sections()
        self._bandpath = None

    @property
    def energy_reference(self):
        return self._energy_reference

    def set_energy_reference(self, reference, soc=False):
        fermi_level = self.fermi_level
        band_extrema = self.band_extrema
        bandgap = self.bandgap

        if type(reference) == str:
            if "mid" in reference.lower():
                reference = "middle"
            elif "fermi" in reference.lower():
                reference = "fermi level"
            elif reference in ["work function", "wf", "vacuum"]:
                reference = "work function"
        elif (type(reference) == float) or (type(reference) == int):
            reference = float(reference)
        else:
            reference = None

        if reference == None:
            if soc:
                metallic = bandgap.soc < 0.1
            else:
                metallic = bandgap.scalar < 0.1
            if metallic:
                # Default for metals is Fermi level.
                reference = "fermi level"
            else:
                # Defaults for insulators is middle.
                reference = "middle"
            if self.work_function != None:
                reference = "work function"

        if reference == "middle":
            logger.debug("Reference energy set to band gap middle.")
            if soc:
                value = (band_extrema.cbm_soc + band_extrema.vbm_soc) / 2
                logger.debug(
                    "The mulliken bands with soc are wrongly referenced to the scalar fermi level. This is a work-around."
                )
                if self.task == "band structure":
                    value -= fermi_level.soc
                elif self.task == "mulliken-projected band structure":
                    value -= fermi_level.scalar
            else:
                value = (band_extrema.cbm_scalar + band_extrema.vbm_scalar) / 2
                value -= fermi_level.scalar
        elif reference == "VBM":
            logger.debug("Reference energy set to valence band maximum.")
            if soc:
                value = band_extrema.vbm_soc - fermi_level.soc
            else:
                value = band_extrema.vbm_scalar - fermi_level.scalar
        elif reference == "work function":
            assert self.work_function != None, "Work function was not calculated."
            logger.debug("Reference energy set to vacuum level.")
            if soc:
                value = -(-self.work_function.upper_vacuum_level + fermi_level.soc)
            else:
                value = -(-self.work_function.upper_vacuum_level + fermi_level.scalar)
        elif reference == "fermi level":
            logger.debug("Reference energy set to Fermi level.")
            # AIMS output is already shifted w.r.t to fermi-level.
            value = 0.0
        elif type(reference) == float:
            value = reference
            reference = "user-specified"
            logger.debug("Reference energy set to {:.4f} eV.".format(value))
        else:
            value = 0.0

        rf = namedtuple("energy_reference", ["reference", "value"])
        self._energy_reference = rf(reference, value)

    def __set_sections(self):
        secs = namedtuple("band_sections", ["regular", "mlk"])
        sec = namedtuple("section", ["k1", "k2", "npoints", "symbol1", "symbol2"])
        s = []
        for i in ["band_sections", "mulliken_band_sections"]:
            sections = self.control[i]
            sections = [k.strip().split() for k in sections]
            sections = [
                sec(
                    np.array(k[2:5], dtype=float),
                    np.array(k[5:8], dtype=float),
                    int(k[8]),
                    k[9],
                    k[10],
                )
                for k in sections
            ]
            s.append(sections)
        return secs(*s)

    @property
    def bandpath(self):
        return self._bandpath

    def set_bandpath(self):
        sections = self.band_sections
        special_points = {k.symbol1: k.k1 for k in sections}
        special_points.update({k.symbol2: k.k2 for k in sections})
        pathstring = [[k.symbol1, k.symbol2] for k in sections]
        rev_string = []
        for i, k in enumerate(pathstring):
            s1, s2 = k
            if i == 0:
                rev_string.append(s1)
                rev_string.append(s2)
            elif s1 == rev_string[-1]:
                rev_string.append(s2)
            else:  # s1 != rev_string[-1]:
                rev_string.append(",")
                rev_string.append(s1)
                rev_string.append(s2)
        pathstring = "".join(rev_string)
        bp = BandPath(
            path=pathstring, cell=self.structure.cell, special_points=special_points
        )
        return bp

    def get_bandpath(self, bandpathstring):
        new_bandpath = parse_path_string(bandpathstring)
        old_path = self.bandpath
        special_points = old_path.special_points
        pairs = [(s.symbol1, s.symbol2) for s in self.band_sections]
        for segment in new_bandpath:
            assert len(segment) > 1, "A vertex needs at least two points."
            p = zip(segment[:-1], segment[1:])
            for s1, s2 in p:
                assert any(
                    x in pairs for x in ((s1, s2), (s2, s1))
                ), "The k-path {}-{} has not been calculated.".format(s1, s2)
        else:
            new_path = BandPath(
                path=bandpathstring,
                cell=self.structure.cell,
                special_points=special_points,
            )
        return new_path

    def get_bandfiles(self, spin="none", soc=False):
        nbf = namedtuple("bandfiles", ["regular", "mulliken"])
        reg = mlk = None
        n = len(self.band_sections)
        if spin in [None, False, "none", "None", 0]:
            spin = "none"
        elif spin in ["up", "UP", "+", 1]:
            spin = "up"
        elif spin in ["dn", "down", "-", 2]:
            spin = "dn"
        if soc and (spin != "none"):
            logger.warning("Collinear spin and SOC are physically not meaningful.")
        if spin != "none":
            assert self.control["spin"] in [
                "collinear"
            ], "Collinear spin was not specified in control.in ."
        if soc:
            assert self.control[
                "include_spin_orbit"
            ], "SOC was not specified in control.in ."
            if "band structure" in self.tasks:
                reg = self.__get_bandfiles_soc(spin=spin)
                assert len(reg) == n, "Wrong number of soc band files found."
            if "mulliken-projected band structure" in self.tasks:
                mlk = self.__get_mlk_bandfiles_soc()
                assert len(mlk) == n, "Wrong number of mlk soc band files found."
        else:
            if "band structure" in self.tasks:
                reg = self.__get_bandfiles_scalar(spin=spin)
                assert len(reg) == n, "Wrong number of scalar band files found."
            if "mulliken-projected band structure" in self.tasks:
                if self.control["include_spin_orbit"]:
                    logger.warning(
                        "Mulliken-projected soc files overwrite scalar files."
                    )
                    mlk = None
                else:
                    mlk = self.__get_mlk_bandfiles_scalar(spin=spin)
                    assert (
                        len(mlk) == n
                    ), "Wrong number of mulliken scalar band files found."
        bandfiles = nbf(reg, mlk)
        return bandfiles

    def __get_bandfiles_scalar(self, spin="none"):
        n = len(self.band_sections)
        files = list(self.outputdir.glob("*.out*"))
        spin = "1" if spin in ["none", "up"] else "2"
        bandfiles = []
        for i in range(1, n + 1):
            soc = self.control["include_spin_orbit"]
            if soc:
                regex = re.compile(
                    r"^band" + spin + "{:03d}".format(i) + r"\.out\.no_soc$"
                )
            else:
                regex = re.compile(r"^band" + spin + "{:03d}".format(i) + r"\.out$")
            f = [j for j in files if bool(regex.match(str(j.parts[-1])))]
            assert (
                len(f) == 1
            ), "Wrong number of band files found for spin = {}, index = {} and soc = {}. Something must have gone wrong.".format(
                spin, i, soc
            )
            bandfiles.append(f[0])
        return bandfiles

    def __get_bandfiles_soc(self, spin="none"):
        n = len(self.band_sections)
        files = list(self.outputdir.glob("*.out*"))
        spin = "1" if spin in ["none", "up"] else "2"
        bandfiles = []
        for i in range(1, n + 1):
            regex = re.compile(r"^band" + spin + "{:03d}".format(i) + r"\.out$")
            f = [j for j in files if bool(regex.match(str(j.parts[-1])))]
            assert (
                len(f) == 1
            ), "Wrong number of band files found for spin = {}, index = {} and soc = {}. Something must have gone wrong.".format(
                spin, i, True
            )
            bandfiles.append(f[0])
        return bandfiles

    def __get_mlk_bandfiles_scalar(self, spin="none"):
        n = len(self.band_sections)
        files = list(self.outputdir.glob("*.out*"))
        spin = "1" if spin in ["none", "up"] else "2"
        bandfiles = []
        for i in range(1, n + 1):
            # mulliken soc files overwrite scalar files
            regex = re.compile(r"^bandmlk" + spin + "{:03d}".format(i) + r"\.out$")
            f = [j for j in files if bool(regex.match(str(j.parts[-1])))]
            assert (
                len(f) == 1
            ), "Wrong number of mlk band files found for spin = {}, index = {} and soc = {}. Something must have gone wrong.".format(
                spin, i, False
            )
            bandfiles.append(f[0])
        return bandfiles

    def __get_mlk_bandfiles_soc(self):
        n = len(self.band_sections)
        files = list(self.outputdir.glob("*.out*"))
        bandfiles = []
        for i in range(1, n + 1):
            regex = re.compile(r"^bandmlk1" + "{:03d}".format(i) + r"\.out$")
            f = [j for j in files if bool(regex.match(str(j.parts[-1])))]
            assert (
                len(f) == 1
            ), "Wrong number of mlk band files found for index = {} and soc = {}. Something must have gone wrong.".format(
                i, True
            )
            bandfiles.append(f[0])
        return bandfiles

    def get_data_from_bandstructure(self, spectrum=None, spin=None):
        from itertools import combinations_with_replacement

        dbg = namedtuple("direct", ["value", "k", "axis_coord", "e1", "e2"])
        ibg = namedtuple(
            "indirect", ["value", "k1", "axis_coord1", "e1", "k2", "axis_coord2", "e2"]
        )

        spin = self.spin2index(spin)

        if spectrum == None:
            spectrum = self.spectrum
        else:
            spectrum = spectrum
        evs = spectrum.eigenvalues[:, spin, :].copy()
        occs = spectrum.occupations[:, spin, :].copy()
        kpts = spectrum.kpoints.copy()
        kcoords = spectrum.kpoint_axis.copy()
        l = range(len(spectrum.kpoints))
        gaps = []
        for i, j in combinations_with_replacement(l, 2):
            vbs = evs[i, :][occs[i, :] >= 1e-4]
            cbs = evs[j, :][occs[j, :] < 1e-4]
            cb = np.min(cbs)
            vb = np.max(vbs)
            gap = cb - vb
            gaps.append((i, j, gap, vb, cb))
        gaps = np.array(gaps)
        index = np.argmin(gaps[:, 2])
        vbm, cbm = np.max(gaps[:, 3]), np.min(gaps[:, 4])
        if gaps[index, 2] > 0.1:
            if gaps[index, 0] != gaps[index, 1]:
                i, j = map(int, gaps[index, [0, 1]])
                kp1 = np.dot(kpts[i], self.structure.cell.T) / (2 * np.pi)
                kp2 = np.dot(kpts[j], self.structure.cell.T) / (2 * np.pi)
                kc1 = kcoords[i]
                kc2 = kcoords[j]
                val = gaps[index, 2]
                indirect = ibg(val, kp1, kc1, vbm, kp2, kc2, cbm)
                dgaps = gaps[gaps[:, 0] == gaps[:, 1]]
                dgaps = dgaps.reshape((-1, 5))
                index = np.argmin(dgaps[:, 2])
                val = dgaps[index, 2]
                i = int(dgaps[index, 0])
                kp = np.dot(kpts[i], self.structure.cell.T) / (2 * np.pi)
                kc = kcoords[i]
                e1, e2 = dgaps[index, 3], dgaps[index, 4]
                direct = dbg(val, kp, kc, e1, e2)
            else:
                i = int(gaps[index, 0])
                val = gaps[index, 2]
                kp = np.dot(kpts[i], self.structure.cell.T) / (2 * np.pi)
                kc = kcoords[i]
                indirect = None
                e1, e2 = gaps[index, 3], gaps[index, 4]
                direct = dbg(val, kp, kc, e1, e2)
        else:
            direct = indirect = None
        return (vbm, cbm, indirect, direct)

    def spin2index(self, spin):
        if spin in [None, "none", "down", "dn", 0]:
            spin = 0
        elif spin in ["up", "UP", 1, 2]:
            spin = 1
        return spin
