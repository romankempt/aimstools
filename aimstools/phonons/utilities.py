from aimstools.misc import *

from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from ase.dft.kpoints import parse_path_string, BandPath
from ase.formula import Formula


from aimstools.density_of_states.utilities import gradient_fill


class PhononSpectrum:
    """Container class for eigenvalue spectrum and associated data.

    Attributes:
        atoms (ase.atoms.Atoms): ASE atoms object.
        qpoints (ndarray): (nqpoints, 3) array with k-points.
        qpoint_axis (ndarray): (nqpoints, 1) linear plotting axis.
        frequencies (ndarray): (nqpoints, nbands) array with frequencies.
        label_coords (list): List of k-point label coordinates on the plotting axis.
        qpoint_labels (list): List of k-point labels.
        jumps (list): List of jumps from unconnected Brillouin zone sections on the plotting axis.
        unit (str): Energy unit.

    """

    def __init__(
        self,
        atoms: "ase.atoms.Atoms" = None,
        qpoints: "numpy.ndarray" = None,
        qpoint_axis: "numpy.ndarray" = None,
        frequencies: "numpy.ndarray" = None,
        label_coords: list = None,
        qpoint_labels: list = None,
        jumps: list = None,
        unit: str = None,
        bandpath: str = None,
    ) -> None:
        self._atoms = atoms
        self._qpoints = qpoints
        self._qpoint_axis = qpoint_axis
        self._frequencies = frequencies
        self._label_coords = label_coords
        self._qpoint_labels = qpoint_labels
        self._jumps = jumps
        self._unit = unit
        self._bandpath = bandpath

    def __repr__(self):
        return "{}(bandpath={}, unit={})".format(
            self.__class__.__name__, self.bandpath, self.unit
        )

    @property
    def atoms(self):
        return self._atoms

    @property
    def qpoints(self):
        return self._qpoints

    @property
    def qpoint_axis(self):
        return self._qpoint_axis

    @property
    def frequencies(self):
        return self._frequencies

    @property
    def label_coords(self):
        return self._label_coords

    @property
    def qpoint_labels(self):
        return self._qpoint_labels

    @property
    def jumps(self):
        return self._jumps

    @property
    def unit(self):
        return self._unit

    @property
    def bandpath(self):
        return self._bandpath


class PhononDOS:
    """Container class for phonon DOS spectrum and associated data.

    Attributes:
        atoms (ase.atoms.Atoms): ASE atoms object.
        frequencies (ndarray): (nqpoints, nbands) array with frequencies.
        label_coords (list): List of k-point label coordinates on the plotting axis.
        qpoint_labels (list): List of k-point labels.
        jumps (list): List of jumps from unconnected Brillouin zone sections on the plotting axis.
        unit (str): Energy unit.

    """

    def __init__(
        self,
        atoms: "ase.atoms.Atoms" = None,
        frequencies: "numpy.ndarray" = None,
        contributions: "numpy.ndarray" = None,
        unit: str = None,
    ) -> None:
        self._atoms = atoms
        self._frequencies = frequencies
        self._contributions = contributions
        self._unit = unit

    def __repr__(self):
        return "{}(unit={})".format(self.__class__.__name__, self.unit)

    @property
    def atoms(self):
        return self._atoms

    @property
    def frequencies(self):
        return self._frequencies

    @property
    def contributions(self):
        return self._contributions

    @property
    def unit(self):
        return self._unit


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

        self.show_acoustic_bands = kwargs.get("show_acoustic_bands", True)
        self.acoustic_bands_color = kwargs.get("acoustic_bands_color", "royalblue")

        self.y_tick_locator = kwargs.get("y_tick_locator", 100)
        self.set_xy_axes_labels()
        self.set_qpoint_labels()
        self.set_x_limits()
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
        self.ax.set_xlim(self.xlimits)
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
        if self.show_acoustic_bands and self.main:
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

    def set_x_limits(self):
        x = self.x
        lower_xlimit = 0.0
        upper_xlimit = np.max(x)
        self.xlimits = (lower_xlimit, upper_xlimit)

    def _show_accoustic(self):
        y = self.y.copy()
        acc = y[:, :3]
        self.ax.plot(
            self.x,
            acc,
            color=self.acoustic_bands_color,
            linewidth=self.bands_linewidth,
            linestyle=self.bands_linestyle,
            alpha=self.bands_alpha,
        )


class PhononDOSPlot:
    """Context to draw Phonon DOS plot. Handles labelling, shifting and broadening."""

    def __init__(
        self,
        main: bool = True,
        dos: "aimstools.phonons.utilities.PhononDOS" = None,
        **kwargs
    ) -> None:
        self.ax = kwargs.get("ax", None)
        assert (
            type(self.ax) != list
        ), "Axes object must be a single matplotlib.axes.Axes, not list."

        self.dos = dos
        self.energies = self.dos.frequencies
        self.contributions = self.dos.contributions
        self.unit = self.dos.unit
        self.flip_axes = kwargs.get("flip_axes", True)
        self.main = main

        self.dos_linewidth = kwargs.get(
            "dos_linewidth", plt.rcParams["lines.linewidth"]
        )
        self.dos_linestyle = kwargs.get("dos_linestyle", "-")

        self.show_grid_lines = kwargs.get("show_grid_lines", False)
        self.grid_lines_axes = kwargs.get("show_grid_lines_axes", "x")
        self.grid_linestyle = kwargs.get("grid_linestyle", (0, (1, 1)))
        self.grid_linewidth = kwargs.get("grid_linewidth", 1.0)
        self.grid_linecolor = kwargs.get("grid_linecolor", mutedblack)

        self.color = kwargs.get("color", mutedblack)

        self.energy_tick_locator = kwargs.get("energy_tick_locator", 100)
        self.dos_tick_locator = kwargs.get("dos_tick_locator", "auto")

        self.fill = kwargs.get("fill", "gradient")

        self.set_dos_window()
        self.set_xy_axes_labels()
        self.set_dos_tick_locator()

    def set_dos_tick_locator(self):
        if self.dos_tick_locator == "auto":
            a, b = self.lower_dos_limit, self.upper_dos_limit
            d = round(abs(b - a) / 3, 1)
            self.dos_tick_locator = d
        else:
            assert isinstance(
                self.dos_tick_locator, (int, float)
            ), "DOS tick locator must be int or float."

    def set_xy_axes_labels(self):
        self.dos_label = r"DOS"
        self.energy_label = "frequency [{}]".format(self.unit)

    def set_dos_window(self):
        tdos = self.contributions.copy()
        self.lower_dos_limit = 0
        self.upper_dos_limit = np.max(tdos) * 1.05

    def draw(self):
        energies = self.energies.copy()
        values = self.contributions.copy()
        if self.flip_axes:
            xlabel = self.dos_label
            ylabel = self.energy_label
            xlimits = (self.lower_dos_limit, self.upper_dos_limit)
            ylimits = (np.min(energies), np.max(energies))
            xlocs = ticker.MultipleLocator(base=self.dos_tick_locator)
            ylocs = ticker.MultipleLocator(base=self.energy_tick_locator)
        else:
            xlabel = self.energy_label
            ylabel = self.dos_label
            ylimits = (np.min(energies), np.max(energies))
            ylimits = (self.lower_dos_limit, self.upper_dos_limit)
            xlocs = ticker.MultipleLocator(base=self.energy_tick_locator)
            ylocs = ticker.MultipleLocator(base=self.dos_tick_locator)

        self.ax.xaxis.set_major_locator(xlocs)
        self.ax.yaxis.set_major_locator(ylocs)
        self.ax.set_xlabel(xlabel, fontsize=plt.rcParams["axes.labelsize"])
        self.ax.set_ylabel(ylabel, fontsize=plt.rcParams["axes.labelsize"])
        self.ax.set_xlim(xlimits)
        self.ax.set_ylim(ylimits)

        if self.flip_axes:
            x = values
            y = energies
        else:
            x = energies
            y = values

        self.ax.plot(
            x,
            y,
            color=self.color,
            linewidth=self.dos_linewidth,
            linestyle=self.dos_linestyle,
        )

        if self.fill == "gradient":
            self.ax = gradient_fill(x, y, self.ax, self.color, flip=self.flip_axes)

        if self.show_grid_lines and self.main:
            self.ax.grid(
                b=self.show_grid_lines,
                which="major",
                axis=self.grid_lines_axes,
                linestyle=self.grid_linestyle,
                linewidth=self.grid_linewidth,
                color=self.grid_linecolor,
            )

