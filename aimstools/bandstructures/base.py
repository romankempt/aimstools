from aimstools.misc import *
from aimstools.postprocessing import FHIAimsOutputReader

from ase.dft.kpoints import bandpath, parse_path_string, BandPath

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
        self._bandpath = None
        self.__set_sections()

    @property
    def energy_reference(self):
        return self._energy_reference

    def set_energy_reference(self, reference, soc=False):
        if not soc:
            vbm, cbm = self.band_extrema[:2]
            bandgap = abs(cbm - vbm)
            fermi_level = self.fermi_level.scalar
        else:
            vbm, cbm = self.band_extrema[2:]
            bandgap = abs(cbm - vbm)
            fermi_level = self.fermi_level.soc

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
            metallic = bandgap < 0.1
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
            shift = (cbm + vbm) / 2 - fermi_level
        elif reference == "VBM":
            logger.debug("Reference energy set to valence band maximum.")
            shift = vbm - fermi_level
        elif reference == "work function":
            assert self.work_function != None, "Work function has not been calculated."
            logger.debug("Reference energy set to vacuum level.")
            logger.warning(
                r"I am not a 100 % sure the work function referencing is done correctly. Please check the relevant equations."
            )
            work_function = self.work_function.upper_work_function
            vacuum_level_upper = work_function + fermi_level
            shift = (
                -(vacuum_level_upper) - fermi_level
            )  # referencing to absolute vacuum
        elif reference == "fermi level":
            logger.debug("Reference energy set to Fermi level.")
            # AIMS output is already shifted w.r.t to fermi-level.
            shift = 0.0
        elif isinstance(reference, (float, tuple)):
            shift = reference
            reference = "user-specified"
            logger.info("Reference energy set to {:.4f} eV.".format(shift))
        else:
            shift = 0.0

        rf = namedtuple("energy_reference", ["reference", "shift"])
        self._energy_reference = rf(reference, shift)

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
        self.band_sections = secs(*s)

    @property
    def bandpath(self):
        return self._bandpath

    def _set_bandpath_from_sections(self):
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
        self._bandpath = bp

    def set_bandpath(self, bandpathstring):
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
        self._bandpath = new_path

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

    def spin2index(self, spin):
        if spin in [None, "none", "down", "dn", 0]:
            spin = 0
        elif spin in ["up", "UP", 1, 2]:
            spin = 1
        return spin
