from aimstools.misc import *
from aimstools.postprocessing.vibes_parser import FHIVibesParser

import yaml
from pathlib import Path
import numpy as np
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from ase.dft.kpoints import parse_path_string, BandPath
from ase.formula import Formula

from collections import namedtuple


class PhononPlot:
    def __init__(
        self, spectrum=None, window=None, accoustic_color=None, main=False
    ) -> None:
        self.ax = plt.gca()
        self.spectrum = spectrum
        self.x = spectrum.qpoint_axis.copy()
        self.y = spectrum.frequencies.copy()
        self.window = window
        self.labels = spectrum.qpoint_labels.copy()
        self.labelcoords = spectrum.label_coords.copy()
        self.jumps = spectrum.jumps.copy()
        self.unit = spectrum.unit
        self.accoustic_color = accoustic_color
        self.main = main

    def draw(self):
        # ylocs = ticker.MultipleLocator(base=0.5)
        # self.ax.yaxis.set_major_locator(ylocs)
        self.ax.set_xlabel("")
        self.ax.set_ylabel("frequency [{}]".format(self.unit))
        labels, coords = self.set_x_labels()
        self.ax.set_xticks(coords)
        self.ax.set_xticklabels(labels)
        self.ax.grid(which="major", axis="x", linestyle=(0, (1, 1)), linewidth=1.0)
        self.ax.tick_params(axis=u"x", which=u"both", length=0)
        if self.main:
            for j in self.jumps:
                self.ax.axvline(
                    x=j,
                    linestyle="-",
                    color=darkgray,
                    linewidth=mpllinewidth,
                )
        self.ax.set_xlim(np.min(self.x), np.max(self.x))
        self.ax.axhline(0, color=mutedblack, alpha=0.8)
        if self.accoustic_color != [None, "none", "None"]:
            self.mark_accoustic()
        return self.ax

    def set_x_labels(self):
        def pretty(kpt):
            if kpt == "G":
                kpt = r"$\Gamma$"
            elif len(kpt) == 2:
                kpt = kpt[0] + "$_" + kpt[1] + "$"
            return kpt

        labels = self.labels
        labels = [pretty(j) for j in labels]
        coords = self.labelcoords
        i = 1
        while i < len(labels):
            if coords[i - 1] == coords[i]:
                labels[i - 1] = labels[i - 1] + "|" + labels[i]
                labels.pop(i)
                coords.pop(i)
            else:
                i += 1

        return labels, coords

    def mark_accoustic(self):
        y = self.y.copy()
        acc = y[:, :3]
        self.y = y[:, 3:]
        self.ax.plot(self.x, acc, color=self.accoustic_color)


class FHIVibesPhonons(FHIVibesParser):
    """Handles phonopy output from FHI-vibes.

    Todo:
        Add DOS.

    """

    def __init__(self, outputdir) -> None:
        super().__init__(outputdir)
        self._get_phonopyfiles()
        self._bandpath = None
        self.bands = self.read_bands()
        self.spectrum = self.get_spectrum()
        # need to think about adding dos

    def __repr__(self):
        return "{}(output directory={})".format(
            self.__class__.__name__, repr(self.outpudir)
        )

    def read_bands(self):
        outputyaml = self.outpudir.joinpath("band.yaml")
        assert outputyaml.exists(), "File band.yaml not found."
        with open(outputyaml, "r") as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise Exception(exc)

        labels = data["labels"]
        bpstring = []
        for i, pair in enumerate(labels):
            if i == 0:
                bpstring.append(pair[0])
            elif i == len(labels) - 1:
                bpstring.append(pair[0])
                bpstring.append(pair[1])
            else:
                bpstring.append(pair[0])
        bpstring = "".join(bpstring)
        bpstring = bpstring.replace("|", ",")
        bplist = parse_path_string(bpstring)

        special_points = (
            self.structure.cell.get_bravais_lattice().bandpath().special_points
        )
        self._bandpath = BandPath(
            path=bpstring, cell=self.structure.cell, special_points=special_points
        )

        phon = data["phonon"]
        nqpoints = data["segment_nqpoint"]

        qpoints = np.array([k["q-position"] for k in phon], dtype=np.float32)
        pbands = np.array(
            [[l["frequency"] for l in k["band"]] for k in phon], dtype=np.float32
        )

        b = namedtuple("band", ["qpoints", "frequencies"])
        count = 0
        bands = {}

        i = 0
        for segment in bplist:
            for pair in zip(segment[:-1], segment[1:]):
                j1, j2 = count, nqpoints[i] + count
                count = j2
                qp = qpoints[j1:j2, :]
                pb = pbands[j1:j2, :]
                bands[pair] = b(qp, pb)
                rev = (pair[1], pair[0])
                bands[rev] = b(qp[::-1], pb[::-1])
                i += 1
        return bands

    def get_bandpath(self, bandpathstring):
        new_bandpath = parse_path_string(bandpathstring)
        old_path = self.bandpath
        special_points = old_path.special_points
        pairs = [(s[0], s[1]) for s in self.bands.keys()]
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

    @property
    def bandpath(self):
        return self._bandpath

    def get_spectrum(self, bandpath=None, unit=r"cm$^{-1}$"):
        assert unit in [r"cm$^{-1}$", "Thz"], "Unit not recognized."
        bands = self.bands
        if bandpath != None:
            bp = parse_path_string(self.get_bandpath(bandpath).path)
        else:
            bp = parse_path_string(self.bandpath.path)
        jumps = []
        qps = []
        qpoint_axis = []
        qpoint_labels = []
        label_coords = []
        spectrum = []
        icell_cv = 2 * np.pi * np.linalg.pinv(self.structure.cell).T
        for segment in bp:
            qpoint_labels.append(segment[0])
            label_coords.append(label_coords[-1] if len(label_coords) > 0 else 0.0)
            for s1, s2 in zip(segment[:-1], segment[1:]):
                freqs = bands[(s1, s2)].frequencies
                qpoints = np.dot(bands[(s1, s2)].qpoints, icell_cv)
                qstep = np.linalg.norm(qpoints[-1, :] - qpoints[0, :])
                qaxis = np.linspace(0, qstep, qpoints.shape[0]) + label_coords[-1]
                qstep += label_coords[-1]
                spectrum.append(freqs)
                qps.append(qpoints)
                qpoint_axis.append(qaxis)
                label_coords.append(qstep)
                qpoint_labels.append(s2)
            jumps.append(label_coords[-1])
        jumps = jumps[:-1]

        spectrum = np.concatenate(spectrum, axis=0)  # unit in Thz
        # 1 Thz = 33.356 cm^-1
        if unit == r"cm$^{-1}$":
            spectrum *= 33.356
        else:
            unit = "Thz"
        qps = np.concatenate(qps, axis=0)
        qpoint_axis = np.concatenate(qpoint_axis, axis=0)
        sp = namedtuple(
            "spectrum",
            [
                "qpoints",
                "qpoint_axis",
                "frequencies",
                "label_coords",
                "qpoint_labels",
                "jumps",
                "unit",
            ],
        )
        return sp(qps, qpoint_axis, spectrum, label_coords, qpoint_labels, jumps, unit)

    def _process_kwargs(self, **kwargs):
        kwargs = kwargs.copy()

        axargs = {}
        axargs["figsize"] = kwargs.pop("figsize", (5, 5))
        axargs["filename"] = kwargs.pop("filename", None)
        axargs["title"] = kwargs.pop("title", None)

        d = {}
        bandpath = kwargs.pop("bandpath", None)

        d["window"] = kwargs.pop("window", 3)
        d["accoustic_color"] = kwargs.pop("mark_accoustic", "royalblue")

        if bandpath != None:
            spectrum = self.get_spectrum(bandpath)
        else:
            spectrum = self.spectrum

        d["spectrum"] = spectrum

        return axargs, kwargs, d

    def plot(self, axes=None, color=mutedblack, main=True, **kwargs):
        axargs, kwargs, bsargs = self._process_kwargs(**kwargs)
        with AxesContext(ax=axes, main=main, **axargs) as axes:
            pbs = PhononPlot(main=main, **bsargs)
            axes = pbs.draw()
            x, y = pbs.x, pbs.y
            axes.plot(x, y, color=color, **kwargs)
        return axes

    def read_thermal_properties(self, unit="per mol"):
        """Reads data from thermal_properties.yaml .

        Regarding units, see: https://gitlab.com/vibes-developers/vibes/-/issues/41

        Note:
            The unit systems of free energy, heat capacity,
            and entropy are kJ/mol, J/K/mol, and J/K/mol, respectively,
            where 1 mol means Na× your input unit cell (not formula unit),
            i.e. you have to divide the value by number of formula unit in your
            unit cell by yourself. For example, in MgO (conventional) unit cell,
            if you want to compare with experimental results in kJ/mol,
            you have to divide the phonopy output by four.

        """

        assert unit in ["per atom", "per mol"], "Unit not recognized."

        outputyaml = self.outpudir.joinpath("thermal_properties.yaml")
        assert outputyaml.exists(), "File thermal_properties.yaml not found."
        with open(outputyaml, "r") as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise Exception(exc)

        ZPE = data["zero_point_energy"]  # kJ/ "mol"
        tp = data["thermal_properties"]
        temperature = np.array([k["temperature"] for k in tp], dtype=np.float32)  # K
        free_energy = np.array(
            [k["free_energy"] for k in tp], dtype=np.float32
        )  # kJ / "mol"
        entropy = np.array([k["entropy"] for k in tp], dtype=np.float32)  # J/K/ "mol"
        heat_capacity = np.array(
            [k["heat_capacity"] for k in tp], dtype=np.float32
        )  # J/K/ "mol"

        d = namedtuple(
            "thermal_properties",
            [
                "temperature",
                "ZPE",
                "free_energy",
                "entropy",
                "heat_capacity",
                "reference",
            ],
        )
        if unit == "per mol":
            cf = self.structure.get_chemical_formula()
            form = Formula(cf).stoichiometry()
            ref = form[1]
            count = form[2]
            ZPE /= count
            free_energy /= count
            entropy /= count
            heat_capacity /= count
            return d(temperature, ZPE, free_energy, entropy, heat_capacity, ref)
        elif unit == "per atom":
            return d(
                temperature,
                ZPE,
                free_energy,
                entropy,
                heat_capacity,
                self.structure.get_chemical_formula(),
            )
        # elif unit == "per cell":
        #     if self.structure.is_2d():
        #         ref = "area"
        #         value = self.structure.cell.copy()[:2, :2]
        #         value = np.abs(np.linalg.det(value))  # per Angström^2
        #         ref = "{:.6f}".format(value) + r" $\AA^{-2}$"
        #     else:
        #         ref = "volume"
        #         value = self.structure.cell.volume  # per Angström^3
        #         ref = "{:.6f}".format(value) + r" $\AA^{-3}$"

        #     from ase.units import mol, kJ

        #     logger.info("Returning units in eV / reference unit.")
        #     fu = kJ / (mol * len(self.structure))
        #     ZPE *= fu / value
        #     free_energy *= fu / value
        #     entropy *= fu / value
        #     heat_capacity *= fu / value
        #     return d(temperature, ZPE, free_energy, entropy, heat_capacity, ref)