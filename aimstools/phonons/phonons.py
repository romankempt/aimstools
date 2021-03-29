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
    """Context to draw phonon plot."""

    def __init__(self, main=True, **kwargs) -> None:
        self.ax = kwargs.get("ax", None)
        assert (
            type(self.ax) != list
        ), "Axes object must be a single matplotlib.axes.Axes, not list."

        self.spectrum = kwargs.get("spectrum", None)
        self.set_data_from_spectrum()

        self.show_grid_lines = kwargs.get("show_grid_lines", True)
        self.grid_lines_axes = kwargs.get("show_grid_lines_axes", "x")
        self.grid_linestyle = kwargs.get("grid_linestyle", (0, (1, 1)))
        self.grid_linewidth = kwargs.get("grid_linewidth", 1.0)
        self.grid_linecolor = kwargs.get("grid_linecolor", mutedblack)

        self.show_jumps = kwargs.get("show_jumps", True)
        self.jumps_linewidth = kwargs.get(
            "jumps_linewidth", plt.rcParams["lines.linewidth"]
        )
        self.jumps_linestyle = kwargs.get("jumps_linestyle", "-")
        self.jumps_linecolor = kwargs.get("jumps_linecolor", mutedblack)

        self.show_bandstructure = kwargs.get("show_bandstructure", True)
        self.bands_color = kwargs.get("bands_color", mutedblack)
        self.bands_color = kwargs.get("color", mutedblack)
        self.bands_linewidth = kwargs.get(
            "bands_linewidth", plt.rcParams["lines.linewidth"]
        )
        self.bands_linewidth = kwargs.get("linewidth", plt.rcParams["lines.linewidth"])
        self.bands_linestyle = kwargs.get("bands_linestyle", "-")
        self.bands_linestyle = kwargs.get("linestyle", "-")
        self.bands_alpha = kwargs.get("bands_alpha", 1.0)
        self.bands_alpha = kwargs.get("alpha", 1.0)

        self.show_accoustic = kwargs.get("show_accoustic", True)
        self.accoustic_bands_color = kwargs.get("accoustic_bands_color", "royalblue")

        self.window = kwargs.get("window", 3)
        self.y_tick_locator = kwargs.get("y_tick_locator", 0.5)
        self.set_xy_axes_labels()
        self.set_qpoint_labels()
        self.main = main

    def set_data_from_spectrum(self):
        spectrum = self.spectrum
        self.labels = spectrum.qpoint_labels.copy()
        self.labelcoords = spectrum.label_coords.copy()
        self.jumps = spectrum.jumps.copy()
        self.x = spectrum.qpoint_axis.copy()
        self.y = spectrum.frequencies.copy()
        self.unit = spectrum.unit

    def draw(self):
        ylocs = ticker.MultipleLocator(base=self.y_tick_locator)
        self.ax.yaxis.set_major_locator(ylocs)
        self.ax.set_xlabel(self.xlabel, fontsize=plt.rcParams["axes.labelsize"])
        self.ax.set_ylabel(self.ylabel, fontsize=plt.rcParams["axes.labelsize"])
        # self.ax.set_xlim(self.xlimits)
        # self.ax.set_ylim(self.ylimits)
        self.ax.set_xticks(self.xlabelcoords)
        self.ax.set_xticklabels(self.xlabels, fontsize=plt.rcParams["axes.labelsize"])
        self.ax.tick_params(axis="x", which="both", length=0)
        if self.show_grid_lines and self.main:
            self.ax.grid(
                b=self.show_grid_lines,
                which="major",
                axis=self.grid_lines_axes,
                linestyle=self.grid_linestyle,
                linewidth=self.grid_linewidth,
                color=self.grid_linecolor,
            )
        if self.show_jumps and self.main:
            for j in self.jumps:
                self.ax.axvline(
                    x=j,
                    linestyle=self.jumps_linestyle,
                    color=self.jumps_linecolor,
                    linewidth=self.jumps_linewidth,
                )
        if self.show_bandstructure and self.main:
            self.ax.plot(
                self.x,
                self.y,
                color=self.bands_color,
                alpha=self.bands_alpha,
                linewidth=self.bands_linewidth,
                linestyle=self.bands_linestyle,
            )
        if self.show_accoustic and self.main:
            self._show_accoustic()

    def set_xy_axes_labels(self):
        self.xlabel = ""
        self.ylabel = "frequency [{}]".format(self.unit)

    def set_qpoint_labels(self):
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

        self.xlabels = labels
        self.xlabelcoords = coords

    def _show_accoustic(self):
        y = self.y.copy()
        acc = y[:, :3]
        self.ax.plot(
            self.x,
            acc,
            color=self.accoustic_color,
            linewidth=self.bands_linewidth,
            linestyle=self.bands_linestyle,
            alpha=self.bands_alpha,
        )


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

    def _process_kwargs(self, kwargs):
        kwargs = kwargs.copy()

        deprecated = ["title", "mark_accoustic"]
        for dep in deprecated:
            if dep in kwargs.keys():
                kwargs.pop(dep)
                logger.warning(
                    f"Keyword {dep} is deprecated. Please do not use this anymore."
                )

        return kwargs

    def plot(self, axes=None, color=mutedblack, main=True, **kwargs):
        kwargs = self._process_kwargs(kwargs)
        bandpath = kwargs.pop("bandpath", None)
        spectrum = self.get_spectrum(bandpath=bandpath)
        with AxesContext(ax=axes, main=main, **kwargs) as axes:
            pbs = PhononPlot(main=main, spectrum=spectrum, **bsargs)
            pbs.draw()
        return axes

    def read_thermal_properties(self, unit="per mol"):
        """Reads data from thermal_properties.yaml .

        Regarding units, see: https://gitlab.com/vibes-developers/vibes/-/issues/41

        Args:
            unit (str): Either "per atom" or "per mol". "per atom" leaves the unit as given out by phonopy, "per mol" divides by the formula count.

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
