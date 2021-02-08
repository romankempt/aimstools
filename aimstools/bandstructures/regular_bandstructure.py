from aimstools.misc import *
from aimstools.bandstructures.base import BandStructureBaseClass
from aimstools.bandstructures.utilities import BandStructurePlot

from ase.dft.kpoints import parse_path_string

from collections import namedtuple

import numpy as np


class RegularBandStructure(BandStructureBaseClass):
    def __init__(self, outputfile, soc=False) -> None:
        super().__init__(outputfile)
        self.soc = soc
        self.spin = "none" if self.control["spin"] != "collinear" else "collinear"
        self.task = "band structure"
        self.band_sections = self.band_sections.regular
        self._bandpath = self.set_bandpath()
        if self.spin == "none":
            bandfiles = self.get_bandfiles(spin="none", soc=soc)
            bandfiles = bandfiles.regular
            self.bands = self.read_bandfiles(zip(bandfiles, bandfiles))
            self.spectrum = self.get_spectrum()
        else:
            if soc:
                logger.warning(
                    "Past Roman to future Roman: Should I just sum up the spin channels in case of SOC?"
                )
            bandfiles_dn = self.get_bandfiles(spin="dn", soc=soc).regular
            bandfiles_up = self.get_bandfiles(spin="up", soc=soc).regular
            self.bands = self.read_bandfiles(zip(bandfiles_dn, bandfiles_up))
            self.spectrum = self.get_spectrum()

    def __repr__(self):
        return "{}(outputfile={}, spin_orbit_coupling={})".format(
            self.__class__.__name__, repr(self.outputfile), self.soc
        )

    def read_bandfiles(self, bandfiles):
        bands = {}
        b = namedtuple("band", ["kpoints", "occupations", "eigenvalues"])
        nspins = 2 if self.spin == "collinear" else 1
        for section, bandfile in zip(self.band_sections, bandfiles):
            ev, occ, rocc, revs = [], [], [], []
            for s in range(nspins):
                bf = bandfile[s]
                # index, k1, k2, k3, occ, ev, occ, ev ...
                data = np.loadtxt(bf)[:, 1:]
                points = data[:, :3]
                occupations = data[:, 3:-2:2]
                eigenvalues = data[:, 4:-1:2]
                rpoints = points[::-1].copy()
                roccs = occupations[::-1].copy()
                reigvs = eigenvalues[::-1].copy()
                ev.append(eigenvalues)
                revs.append(reigvs)
                occ.append(occupations)
                rocc.append(roccs)
            ev = np.stack(ev, axis=1)
            occ = np.stack(occ, axis=1)
            rocc = np.stack(rocc, axis=1)
            revs = np.stack(revs, axis=1)
            band_forward = b(points, occ, ev)
            band_backward = b(rpoints, rocc, revs)
            pathsegment = (section.symbol1, section.symbol2)
            pathsegment_r = (section.symbol2, section.symbol1)
            bands[pathsegment] = band_forward
            bands[pathsegment_r] = band_backward
        return bands

    def get_spectrum(self, bandpath=None):
        bands = self.bands
        if bandpath != None:
            bp = parse_path_string(self.get_bandpath(bandpath).path)
        else:
            bp = parse_path_string(self.bandpath.path)
        jumps = []
        kps = []
        occs = []
        kpoint_axis = []
        kpoint_labels = []
        label_coords = []
        spectrum = []
        icell_cv = 2 * np.pi * np.linalg.pinv(self.structure.cell).T
        for segment in bp:
            kpoint_labels.append(segment[0])
            label_coords.append(label_coords[-1] if len(label_coords) > 0 else 0.0)
            for s1, s2 in zip(segment[:-1], segment[1:]):
                energies = bands[(s1, s2)].eigenvalues
                kpoints = np.dot(bands[(s1, s2)].kpoints, icell_cv)
                occ = bands[(s1, s2)].occupations
                kstep = np.linalg.norm(kpoints[-1, :] - kpoints[0, :])
                kaxis = np.linspace(0, kstep, kpoints.shape[0]) + label_coords[-1]
                kstep += label_coords[-1]
                spectrum.append(energies)
                kps.append(kpoints)
                occs.append(occ)
                kpoint_axis.append(kaxis)
                label_coords.append(kstep)
                kpoint_labels.append(s2)
            jumps.append(label_coords[-1])
        jumps = jumps[:-1]

        spectrum = np.concatenate(spectrum, axis=0)
        kps = np.concatenate(kps, axis=0)
        kpoint_axis = np.concatenate(kpoint_axis, axis=0)
        occs = np.concatenate(occs, axis=0)
        sp = namedtuple(
            "spectrum",
            [
                "kpoints",
                "kpoint_axis",
                "eigenvalues",
                "occupations",
                "label_coords",
                "kpoint_labels",
                "jumps",
            ],
        )
        return sp(kps, kpoint_axis, spectrum, occs, label_coords, kpoint_labels, jumps)

    def _process_kwargs(self, **kwargs):
        kwargs = kwargs.copy()

        axargs = {}
        axargs["figsize"] = kwargs.pop("figsize", (5, 5))
        axargs["filename"] = kwargs.pop("filename", None)
        axargs["title"] = kwargs.pop("title", None)

        d = {}
        bandpath = kwargs.pop("bandpath", None)
        spin = kwargs.pop("spin", None)
        reference = kwargs.pop("reference", None)

        d["window"] = kwargs.pop("window", 3)
        d["mark_fermi_level"] = kwargs.pop("mark_fermi_level", fermi_color)
        d["mark_gap"] = kwargs.pop("mark_gap", True)

        self.set_energy_reference(reference, self.soc)
        if bandpath != None:
            spectrum = self.get_spectrum(bandpath)
        else:
            spectrum = self.spectrum
        vbm, cbm, indirect_gap, direct_gap = self.get_data_from_bandstructure(
            spectrum, spin=spin
        )
        ref, shift = self.energy_reference
        fermi_level = self.fermi_level.soc if self.soc else self.fermi_level.scalar

        d["spectrum"] = spectrum
        d["spin"] = self.spin2index(spin)
        d["vbm"] = vbm
        d["cbm"] = cbm
        d["indirect_gap"] = indirect_gap
        d["direct_gap"] = direct_gap
        d["ref"] = ref
        d["shift"] = shift
        d["fermi_level"] = fermi_level

        return axargs, kwargs, d

    def plot(self, axes=None, color=mutedblack, main=True, **kwargs):
        axargs, kwargs, bsargs = self._process_kwargs(**kwargs)
        with AxesContext(ax=axes, main=main, **axargs) as axes:
            bs = BandStructurePlot(main=main, **bsargs)
            axes = bs.draw()
            x, y = bs.xy
            axes.plot(x, y, color=color, **kwargs)
        return axes
