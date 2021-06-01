from aimstools.misc import *

import numpy as np
from scipy.integrate import trapz

import matplotlib.pyplot as plt
from matplotlib.path import Path as mplPath
from matplotlib.patches import PathPatch
import matplotlib.colors
import matplotlib.ticker as ticker

from ase.symbols import string2symbols
from ase.data import chemical_symbols
from ase.formula import Formula
from matplotlib.lines import Line2D


def delta_function(energies, energy, width):
    """Return a delta-function centered at 'energy'."""
    x = -(((energies - energy) / width) ** 2)
    d = np.exp(x) / (np.sqrt(np.pi) * width)
    return d


def smear_dos(energies, dos, width=0.1):
    """ Broadens DOS by a delta function while maintaining the same area."""
    new_dos = np.zeros(dos.shape)
    oldI = trapz(dos, energies)
    for e, d in zip(energies, dos):
        new_dos += d * delta_function(energies, e, width)
    newI = trapz(new_dos, energies)
    new_dos *= oldI / newI
    return new_dos


def gradient_fill(x, y, axes, color, flip=True):
    """
    Plot a linear alpha gradient beneath x y values.
    Here, x and y are transposed due to the nature of DOS graphs.

    Parameters
    ----------
    x, y : array-like
        The data values of the line.
    Additional arguments are passed on to matplotlib's ``plot`` function.
    """

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    xy = np.column_stack([x, y])
    if flip:
        z = np.empty((1, 100, 4), dtype=float)
        z[:, :, -1] = np.linspace(0, 1, 100)[None, :]
        rgb = matplotlib.colors.colorConverter.to_rgb(color)
        z[:, :, :3] = rgb
        im = axes.imshow(
            z, aspect="auto", extent=[xmin, xmax, ymin, ymax], origin="upper"
        )
        path = np.vstack([[0, ymin], xy, [0, ymax], [0, 0], [0, 0]])
    else:
        z = np.empty((100, 1, 4), dtype=float)
        z[:, :, -1] = np.linspace(0, 1, 100)[:, None]
        rgb = matplotlib.colors.colorConverter.to_rgb(color)
        z[:, :, :3] = rgb
        im = axes.imshow(
            z, aspect="auto", extent=[xmin, xmax, ymin, ymax], origin="lower"
        )
        path = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    path = mplPath(path, closed=True)
    patch = PathPatch(path, facecolor="none", edgecolor="none")
    axes.add_patch(patch)
    im.set_clip_path(patch)
    return axes


class DOSSpectrum:
    """Container class for density of states and associated data."""

    def __init__(
        self,
        atoms: "ase.atoms.Atoms" = None,
        energies: "numpy.ndarray" = None,
        contributions: list = None,
        type: str = None,
        fermi_level: float = None,
        band_extrema: tuple = None,
        reference: str = None,
        shift: float = None,
    ):
        self._atoms = atoms
        self._energies = energies
        self._contributions = contributions
        self._type = type
        self._fermi_level = fermi_level
        self._reference = reference
        self._band_extrema = band_extrema
        self._shift = shift

    @property
    def atoms(self):
        return self._atoms

    @property
    def type(self):
        "Classifies if spectrum contains species- or atom-projection or just total DOS."
        return self._type

    @property
    def energies(self):
        "Energies in eV as (nspins, nenergies) array."
        return self._energies

    @property
    def contributions(self):
        "List of (index, ndarray) or (symbol, ndarray)."
        return self._contributions

    @property
    def fermi_level(self):
        "Fermi level in eV."
        return self._fermi_level

    @property
    def reference(self):
        "Reference energy description."
        return self._reference

    @property
    def band_extrema(self):
        "Band extrema (VBM, CBM) from FHI-aims output file."
        return self._band_extrema

    @property
    def shift(self):
        "Shift value to reference energy."
        return self._shift

    def __repr__(self) -> str:
        return "{}(type={})".format(self.__class__.__name__, self.type)

    def get_total_dos(self):
        "Returns total sum of contributions."
        con = sum([k for i, k in self.contributions])
        symbol = self.atoms.get_chemical_formula()
        values = con[:, :, 0]
        return DOSContribution(symbol, values, l=0)

    def get_atom_contribution(self, index, l="tot"):
        "Returns :class:`~aimstools.density_of_states.utilities.DOSContribution` of given atom index and angular momentum l."
        assert (
            self.type == "atom"
        ), "This spectrum type does not support atom contributions."
        l = self._l2index(l)
        symbol = self.atoms[index].symbol
        con = [k for i, k in self.contributions if i == index][0]
        con = con[:, :, l]
        return DOSContribution(symbol, con, l)

    def get_species_contribution(self, symbol, l="tot"):
        """Returns sum of contributions for given species symbol and angular momentum l."""
        l = self._l2index(l)
        if self.type == "atom":
            indices = [i for i, s in enumerate(self.atoms) if s.symbol == symbol]
            con = sum([k for i, k in self.contributions if i in indices])
            con = con[:, :, l]
        elif self.type == "species":
            con = [k for i, k in self.contributions if i == symbol][0]
            con = con[:, :, l]
        else:
            raise Exception(
                "This spectrum type does not support species contributions."
            )
        return DOSContribution(symbol, con, l)

    def get_group_contribution(self, symbols, l="tot"):
        """Returns sum of :class:`~aimstools.bandstructures.mulliken_bandstructure.MullikenContribution` of given list of species."""
        symbols = [string2symbols(s) for s in symbols]
        symbols = set([item for sublist in symbols for item in sublist])
        cons = sum([self.get_species_contribution(s, l) for s in symbols])
        return cons

    def _l2index(self, l):
        if l in [None, "none", "None", "total", "tot"]:
            return 0
        elif l in ["s", 0]:
            return 1
        elif l in ["p", 1]:
            return 2
        elif l in ["d", 2]:
            return 3
        elif l in ["f", 3]:
            return 4
        elif l in ["g", 4]:
            return 5
        elif l in ["h", 5]:
            return 6
        else:
            raise Exception("Implemented DOS contributions only till h-orbitals.")


class DOSContribution:
    """Container class for density of states contributions.

    Supports addition."""

    def __init__(self, symbol, values, l="total") -> None:
        self._symbol = symbol
        self._values = values
        self._l = l

    def __repr__(self) -> str:
        return "{}({})".format(self.__class__.__name__, self.symbol)

    def __add__(self, other) -> "DOSContribution":
        assert (
            self.values.shape == other.values.shape
        ), "DOS contributions shape does not match for addition."
        d = self.values + other.values
        l = "".join(set([self.l, other.l]))
        s1 = string2symbols(self.symbol)
        s2 = string2symbols(other.symbol)
        s = Formula.from_list(s1 + s2).format("reduce").format("metal")
        return DOSContribution(s, d, l)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    @property
    def symbol(self):
        """Symbol to describe atomic species."""
        return self._symbol

    @property
    def values(self):
        """Contribution values as (nenergies, nspins, 7) array."""
        return self._values

    def set_symbol(self, symbol):
        assert type(symbol) == str, "Symbol must be a string."
        try:
            s = string2symbols(symbol)
        except Exception as expt:
            raise Exception("String could not be interpreted as atomic symbols.")
        assert all(
            k in chemical_symbols for k in s
        ), "Symbol is not an element from the PSE."
        s = Formula.from_list(s).format("reduce").format("metal")
        self._symbol = s

    def get_latex_symbol(self):
        s = self.symbol
        s = Formula(s).format("latex")
        if self.l != "total":
            s += f"$_{self.l}$"
        return s

    @property
    def l(self):
        """Angular momentum."""
        self._l = self._index2l(self._l)
        return self._l

    def _index2l(self, l):
        if isinstance(l, (int,)):
            if l == 0:
                return "total"
            if l == 1:
                return "s"
            if l == 2:
                return "p"
            if l == 3:
                return "d"
            if l == 4:
                return "f"
            if l == 5:
                return "g"
            if l == 6:
                return "h"
        elif isinstance(l, (str,)):
            return l
        else:
            raise Exception("Angular momentum not recognized.")


class DOSPlot:
    """Context to draw DOS plot. Handles labelling, shifting and broadening."""

    def __init__(self, main: bool = True, contributions: list = None, **kwargs) -> None:
        self.ax = kwargs.get("ax", None)
        assert (
            type(self.ax) != list
        ), "Axes object must be a single matplotlib.axes.Axes, not list."

        self.spectrum = kwargs.get("spectrum", None)
        self.contributions = contributions
        self.spin = kwargs.get("spin", 0)
        self.spin_factor = 1 if self.spin == 0 else -1
        self.flip_axes = kwargs.get("flip_axes", True)
        self.set_data_from_spectrum()
        self.main = main

        self.dos_linewidth = kwargs.get(
            "dos_linewidth", plt.rcParams["lines.linewidth"]
        )
        self.dos_linestyle = kwargs.get("dos_linestyle", "-")

        self.show_fermi_level = kwargs.get("show_fermi_level", True)
        self.fermi_level_color = kwargs.get("fermi_level_color", fermi_color)
        self.fermi_level_alpha = kwargs.get("fermi_level_alpha", 1.0)
        self.fermi_level_linestyle = kwargs.get("fermi_level_linestyle", "--")
        self.fermi_level_linewidth = kwargs.get(
            "fermi_level_linewidth", plt.rcParams["lines.linewidth"]
        )

        self.show_grid_lines = kwargs.get("show_grid_lines", False)
        self.grid_lines_axes = kwargs.get("show_grid_lines_axes", "x")
        self.grid_linestyle = kwargs.get("grid_linestyle", (0, (1, 1)))
        self.grid_linewidth = kwargs.get("grid_linewidth", 1.0)
        self.grid_linecolor = kwargs.get("grid_linecolor", mutedblack)

        self.colors = kwargs.get("colors", ["red", "blue", "green"])
        self.labels = kwargs.get("labels", [1, 2, 3])

        self.show_legend = kwargs.get("show_legend", True)
        self.legend_linewidth = kwargs.get(
            "legend_linewidth", plt.rcParams["lines.linewidth"]
        )
        self.legend_frameon = kwargs.get(
            "legend_frameon", plt.rcParams["legend.frameon"]
        )
        self.legend_fancybox = kwargs.get(
            "legend_fancybox", plt.rcParams["legend.fancybox"]
        )
        self.legend_borderpad = kwargs.get(
            "legend_borderpad", plt.rcParams["legend.borderpad"]
        )
        self.legend_loc = kwargs.get("legend_loc", "upper right")
        self.legend_handlelength = kwargs.get(
            "legend_handlelength", plt.rcParams["legend.handlelength"]
        )

        self.window = kwargs.get("window", 3)
        self.energy_tick_locator = kwargs.get("energy_tick_locator", 0.5)
        self.dos_tick_locator = kwargs.get("dos_tick_locator", "auto")

        self.broadening = kwargs.get("broadening", 0.0)
        self.fill = kwargs.get("fill", "gradient")

        self.show_total_dos = kwargs.get("show_total_dos", True)
        self.total_dos_linestyle = kwargs.get("total_dos_linestyle", (0, (1, 1)))
        self.total_dos_linewidth = kwargs.get(
            "total_dos_linewidth", plt.rcParams["lines.linewidth"]
        )
        self.total_dos_color = kwargs.get("color", mutedblack)
        self.total_dos_color = kwargs.get("total_dos_color", mutedblack)

        self.set_total_dos()
        self.set_energy_window()
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

    def set_data_from_spectrum(self):
        spectrum = self.spectrum
        self.reference = spectrum.reference
        self.fermi_level = spectrum.fermi_level
        self.shift = spectrum.shift
        self.energies = (spectrum.energies[:, self.spin].copy()) + self.shift
        self.fermi_level = spectrum.fermi_level
        self.band_extrema = spectrum.band_extrema

    def draw(self):
        energies = self.energies.copy()
        if self.flip_axes:
            xlabel = self.dos_label
            ylabel = self.energy_label
            xlimits = (self.lower_dos_limit, self.upper_dos_limit)
            ylimits = (self.lower_energy_limit, self.upper_energy_limit)
            xlocs = ticker.MultipleLocator(base=self.dos_tick_locator)
            ylocs = ticker.MultipleLocator(base=self.energy_tick_locator)
        else:
            xlabel = self.energy_label
            ylabel = self.dos_label
            xlimits = (self.lower_energy_limit, self.upper_energy_limit)
            ylimits = (self.lower_dos_limit, self.upper_dos_limit)
            xlocs = ticker.MultipleLocator(base=self.energy_tick_locator)
            ylocs = ticker.MultipleLocator(base=self.dos_tick_locator)

        self.ax.xaxis.set_major_locator(xlocs)
        self.ax.yaxis.set_major_locator(ylocs)
        self.ax.set_xlabel(xlabel, fontsize=plt.rcParams["axes.labelsize"])
        self.ax.set_ylabel(ylabel, fontsize=plt.rcParams["axes.labelsize"])
        self.ax.set_xlim(xlimits)
        self.ax.set_ylim(ylimits)

        handles = []
        for i, con in enumerate(self.contributions):
            values = con.values[:, self.spin] * self.spin_factor
            label = self.labels[i]
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=self.colors[i],
                    label=label,
                    lw=self.legend_linewidth,
                )
            )
            if self.broadening > 0.0:
                values = smear_dos(energies, values, width=self.broadening)
            if self.flip_axes:
                x = values
                y = energies
            else:
                x = energies
                y = values

            self.ax.plot(
                x,
                y,
                color=self.colors[i],
                linewidth=self.dos_linewidth,
                linestyle=self.dos_linestyle,
            )

            if self.fill == "gradient":
                self.ax = gradient_fill(
                    x, y, self.ax, self.colors[i], flip=self.flip_axes
                )
        if self.show_fermi_level and self.main:
            self._show_fermi_level()
        if self.show_grid_lines and self.main:
            self.ax.grid(
                b=self.show_grid_lines,
                which="major",
                axis=self.grid_lines_axes,
                linestyle=self.grid_linestyle,
                linewidth=self.grid_linewidth,
                color=self.grid_linecolor,
            )
        if self.show_total_dos and self.main:
            self._show_total_dos()
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=self.total_dos_color,
                    label="total",
                    linestyle=self.total_dos_linestyle,
                    lw=self.legend_linewidth,
                )
            )
        if self.show_legend and self.main:
            self._show_legend(handles)

    def set_energy_window(self):
        window = self.window
        if isinstance(window, (float, int)):
            lower_limit, upper_limit = (-window, window)
            if self.reference in ["work function", "user-specified", "vacuum"]:
                lower_limit, upper_limit = (-window + self.shift, window + self.shift)
        elif len(window) == 2:
            lower_limit, upper_limit = window[0], window[1]
        else:
            raise Exception("Window not recognized.")
        self.lower_energy_limit = lower_limit
        self.upper_energy_limit = upper_limit

    def set_xy_axes_labels(self):
        if self.reference in ["fermi level", "VBM", "middle"]:
            energy_label = r"E - E$_{\mathrm{F}}$ [eV]"
        elif self.reference == "vacuum":
            energy_label = r"E - E$_{vacuum}$ [eV]"
        else:
            energy_label = r"E [eV]"
        self.dos_label = r"DOS [states/eV]"
        self.energy_label = energy_label

    def set_dos_window(self):
        # I'm always setting the limits relative to the total dos
        lower_elimit, upper_elimit = self.lower_energy_limit, self.upper_energy_limit
        energies = self.energies.copy()
        tdos = self.total_dos
        values = tdos.values[:, self.spin].copy() * self.spin_factor
        # assert that DOS either starts at 0 or a negative peak for spin
        lower_dos_limit = np.min(
            values[(energies >= lower_elimit) & (energies <= upper_elimit)]
        )
        if tdos.values.shape[1] == 2:  # check if there is a second spin channel
            self.lower_dos_limit = -min([-lower_dos_limit, lower_dos_limit]) * 1.05
        else:
            self.lower_dos_limit = min([0, lower_dos_limit]) * 1.05
        upper_dos_limit = np.max(
            values[(energies >= lower_elimit) & (energies <= upper_elimit)]
        )
        if tdos.values.shape[1] == 2:  # check if there is a second spin channel
            self.upper_dos_limit = max([-upper_dos_limit, upper_dos_limit]) * 1.05
        else:
            self.upper_dos_limit = max([0, upper_dos_limit]) * 1.05

    def set_total_dos(self):
        tdos = self.spectrum.get_total_dos()
        self.total_dos = tdos

    def _show_fermi_level(self):
        reference = self.spectrum.reference
        value = self.spectrum.shift
        if reference in ["user-specified"]:
            mark = value
        elif reference in ["vacuum"]:
            mark = value + (self.band_extrema[0] - self.fermi_level)
        else:
            mark = 0.00

        if self.flip_axes:
            self.ax.axhline(
                y=mark,
                color=self.fermi_level_color,
                alpha=self.fermi_level_alpha,
                linestyle=self.fermi_level_linestyle,
                linewidth=self.fermi_level_linewidth,
            )
        else:
            self.ax.axvline(
                x=mark,
                color=self.fermi_level_color,
                alpha=self.fermi_level_alpha,
                linestyle=self.fermi_level_linestyle,
                linewidth=self.fermi_level_linewidth,
            )

    def _show_total_dos(self):
        tdos = self.total_dos
        values = tdos.values[:, self.spin].copy() * self.spin_factor
        energies = self.energies.copy()
        if self.broadening > 0.0:
            values = smear_dos(energies, values, self.broadening)
        if self.flip_axes:
            x = values
            y = energies
        else:
            x = energies
            y = values
        self.ax.plot(
            x,
            y,
            color=self.total_dos_color,
            linestyle=self.total_dos_linestyle,
            linewidth=self.total_dos_linewidth,
        )

    def _show_legend(self, handles):
        self.ax.legend(
            handles=handles,
            frameon=self.legend_frameon,
            fancybox=self.legend_fancybox,
            borderpad=self.legend_borderpad,
            loc=self.legend_loc,
            handlelength=self.legend_handlelength,
            fontsize=plt.rcParams["legend.fontsize"],
        )

