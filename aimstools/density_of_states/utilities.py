from aimstools.misc import *

import numpy as np
from scipy.integrate import trapz

import matplotlib.pyplot as plt
from matplotlib.path import Path as mplPath
from matplotlib.patches import PathPatch
import matplotlib.colors
import matplotlib.ticker as ticker

from ase.symbols import Symbols, string2symbols, symbols2numbers
from ase.data import chemical_symbols, atomic_masses
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


class DOSPlot:
    """Context to draw DOS plot. Handles labelling, shifting and broadening."""

    def __init__(self, main=True, **kwargs) -> None:
        self.ax = kwargs.get("ax", None)
        assert (
            type(self.ax) != list
        ), "Axes object must be a single matplotlib.axes.Axes, not list."

        self.spectrum = kwargs.get("spectrum", None)
        self.spin = kwargs.get("spin", 0)
        self.spin_factor = 1 if self.spin == 0 else -1
        self.angular_momentum = self.l2index(kwargs.get("l", "tot"))
        self.flip_axes = kwargs.get("flip_axes", True)
        self.set_data_from_spectrum()
        self.main = main

        self.dos_linewidth = kwargs.get("dos_linewidth", mpllinewidth)
        self.dos_linestyle = kwargs.get("dos_linestyle", "-")

        self.show_fermi_level = kwargs.get("show_fermi_level", True)
        self.fermi_level_color = kwargs.get("fermi_level_color", fermi_color)
        self.fermi_level_alpha = kwargs.get("fermi_level_alpha", 1.0)
        self.fermi_level_linestyle = kwargs.get("fermi_level_linestyle", "--")
        self.fermi_level_linewidth = kwargs.get("fermi_level_linewidth", mpllinewidth)

        self.show_grid_lines = kwargs.get("show_grid_lines", True)
        self.grid_lines_axes = kwargs.get("show_grid_lines_axes", "x")
        self.grid_linestyle = kwargs.get("grid_linestyle", (0, (1, 1)))
        self.grid_linewidth = kwargs.get("grid_linewidth", 1.0)

        self.colors = kwargs.get("colors", ["red", "blue", "green"])
        self.labels = kwargs.get("labels", [1, 2, 3])

        self.show_legend = kwargs.get("show_legend", True)
        self.legend_linewidth = kwargs.get("legend_linewidth", 1.5)
        self.legend_frameon = kwargs.get("legend_frameon", True)
        self.legend_fancybox = kwargs.get("legend_fancybox", True)
        self.legend_borderpad = kwargs.get("legend_borderpad", 0.4)
        self.legend_loc = kwargs.get("legend_loc", "upper right")

        self.window = kwargs.get("window", 3)
        self.energy_tick_locator = kwargs.get("energy_tick_locator", 0.5)

        self.broadening = kwargs.get("broadening", 0.0)
        self.fill = kwargs.get("fill", "gradient")

        self.show_total_dos = kwargs.get("show_total_dos", True)
        self.total_dos_linestyle = kwargs.get("total_dos_linestyle", (0, (1, 1)))
        self.total_dos_linewidth = kwargs.get("total_dos_linewidth", mpllinewidth)
        self.total_dos_color = kwargs.get("color", mutedblack)
        self.total_dos_color = kwargs.get("total_dos_color", mutedblack)

        self.set_total_dos()
        self.set_energy_window()
        self.set_dos_window()
        self.set_xy_axes_labels()

    def set_data_from_spectrum(self):
        spectrum = self.spectrum
        self.reference = spectrum.reference
        self.fermi_level = spectrum.fermi_level
        self.shift = spectrum.shift
        self.energies = (spectrum.energies[:, self.spin].copy()) - self.shift
        self.contributions = spectrum.contributions
        self.fermi_level = spectrum.fermi_level

    def draw(self):
        energies = self.energies.copy()
        if self.flip_axes:
            xlabel = self.dos_label
            ylabel = self.energy_label
            xlimits = (self.lower_dos_limit, self.upper_dos_limit)
            ylimits = (self.lower_energy_limit, self.upper_energy_limit)
            xlocs = ticker.MultipleLocator(base=1.0)
            ylocs = ticker.MultipleLocator(base=self.energy_tick_locator)
        else:
            xlabel = self.energy_label
            ylabel = self.dos_label
            xlimits = (self.lower_energy_limit, self.upper_energy_limit)
            ylimits = (self.lower_dos_limit, self.upper_dos_limit)
            xlocs = ticker.MultipleLocator(base=self.energy_tick_locator)
            ylocs = ticker.MultipleLocator(base=1.0)

        self.ax.xaxis.set_major_locator(xlocs)
        self.ax.yaxis.set_major_locator(ylocs)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlim(xlimits)
        self.ax.set_ylim(ylimits)

        handles = []
        for i, con in enumerate(self.contributions):
            values = con.values[:, self.spin, self.angular_momentum] * self.spin_factor
            latex_symbol = con.get_latex_symbol()
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=self.colors[i],
                    label=latex_symbol,
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
        if self.show_total_dos and self.main:
            self._show_total_dos()
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=self.total_dos_color,
                    label="total",
                    linestyle=self.total_dos_linestyle,
                    lw=self.total_dos_linewidth,
                )
            )
        if self.show_legend and self.main:
            self._show_legend(handles)

    def set_energy_window(self):
        window = self.window
        if isinstance(window, (float, int)):
            lower_limit, upper_limit = (-window, window)
            if self.reference in ["work function", "user-specified"]:
                lower_limit, upper_limit = (-window - self.shift, window - self.shift)
        elif len(window) == 2:
            lower_limit, upper_limit = window[0], window[1]
        else:
            raise Exception("Window not recognized.")
        self.lower_energy_limit = lower_limit
        self.upper_energy_limit = upper_limit

    def set_xy_axes_labels(self):
        if self.reference in ["fermi level", "VBM", "middle"]:
            energy_label = r"E - E$_{\mathrm{F}}$ [eV]"
        elif self.reference == "work function":
            energy_label = r"E - E$_{vacuum}$ [eV]"
        else:
            energy_label = r"E [eV]"
        self.dos_label = r"DOS [states/eV]"
        self.energy_label = energy_label

    def set_dos_window(self):
        # I'm always setting the limits relative to the total dos
        lower_elimit, upper_elimit = self.lower_energy_limit, self.upper_energy_limit
        energies = self.energies.copy()
        tdos = self.total_dos.copy()
        # assert that DOS either starts at 0 or a negative peak for spin
        lower_dos_limit = np.min(
            tdos[(energies >= lower_elimit) & (energies <= upper_elimit)]
        )
        self.lower_dos_limit = min([0, lower_dos_limit]) * 1.05
        upper_dos_limit = np.max(
            tdos[(energies >= lower_elimit) & (energies <= upper_elimit)]
        )
        self.upper_dos_limit = max([0, upper_dos_limit]) * 1.05

    def set_total_dos(self):
        tdos = self.spectrum.get_total_dos()
        values = tdos.values[:, self.spin, 0] * self.spin_factor
        energies = self.energies.copy()
        if self.broadening > 0.0:
            values = smear_dos(energies, values, self.broadening)
        self.total_dos = values

    def _show_fermi_level(self):
        reference = self.reference
        value = self.shift
        if reference in ["work function", "user-specified"]:
            mark = -value
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
        energies = self.energies.copy()
        if self.flip_axes:
            x = tdos
            y = energies
        else:
            x = energies
            y = tdos
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
        )

    def l2index(self, l):
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
            raise Exception("Angular momentum l cannot be higher than 5.")

    @property
    def handles(self):
        return self._handles


class DOSSpectrum:
    """Container class for density of states and associated data."""

    def __init__(
        self,
        energies: "numpy.ndarray" = None,
        contributions: "aimstools.density_of_states.utilities.DOSContribution" = None,
        type: str = None,
        fermi_level: float = None,
        reference: str = None,
        shift: float = None,
    ):
        self._energies = energies
        self._contributions = contributions
        self._type = type
        self._fermi_level = fermi_level
        self._reference = reference
        self._shift = shift

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
        "List of :class:~`aimstools.density_of_states.utilities.DOSContribution`."
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
    def shift(self):
        "Shift value to reference energy."
        return self._shift

    def __repr__(self) -> str:
        return "{}(type={})".format(self.__class__.__name__, self.type)

    def get_total_dos(self):
        "Returns total sum of contributions."
        tdos = sum([k for k in self.contributions])
        return tdos

    def get_species_contribution(self, symbol):
        """Returns sum of contributions for given species symbol."""
        if self.type == "species":
            con = [k for k in self.contributions if k.symbol == symbol][0]
        elif self.type == "atom":
            con = sum([k for k in self.contributions if k.symbol == symbol])
        else:
            raise Exception(
                "This spectrum type does not support species contributions."
            )
        return con

    def get_atom_contribution(self, index):
        "Returns :class:~`aimstools.density_of_states.utilities.DOSContribution` of given atom index."
        assert (
            self.type == "atom"
        ), "This spectrum type does not support atom contributions."
        con = self.contributions[index]
        return con


class DOSContribution:
    """Container class for density of states contributions.

    Supports addition."""

    def __init__(self, symbol, values) -> None:
        self._symbol = symbol
        self._values = values

    def __repr__(self) -> str:
        return "{}({})".format(self.__class__.__name__, self.symbol)

    def __add__(self, other) -> "DOSContribution":
        d = self.values + other.values
        s1 = string2symbols(self.symbol)
        s2 = string2symbols(other.symbol)
        s = Formula.from_list(s1 + s2).format("reduce").format("metal")
        return DOSContribution(s, d)

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
        s = Formula(s)
        return s.format("latex")
