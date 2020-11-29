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
    new_dos = np.zeros(dos.shape)
    oldI = trapz(dos, energies)
    for e, d in zip(energies, dos):
        new_dos += d * delta_function(energies, e, width)
    newI = trapz(new_dos, energies)
    new_dos *= oldI / newI
    return new_dos


class DOSPlot:
    def __init__(
        self,
        x=None,
        con=None,
        l="tot",
        color=mutedblack,
        spin=None,
        ref=None,
        shift=None,
        window=None,
        vbm=None,
        cbm=None,
        fermi_level=None,
        mark_fermi_level=None,
        mark_gap=False,
        broadening=0.0,
        fill=None,
        flip=True,
        show_total=[],
        main=True,
    ) -> None:
        self.ax = plt.gca()
        self.flip = flip
        self.broadening = broadening
        l = self.l2index(l)
        # spin up is positive, spin down is negative
        s = 1 if spin == 0 else -1
        x = x[:, spin].copy() - shift  # energies
        self.x = x.copy()
        self.con = con
        y = self.con.values
        self.y = y[:, spin, l].copy() * s
        self.spin = spin
        self.ref = ref
        self.shift = shift
        self.window = self.set_window(window)
        self.fermi_level = fermi_level
        self.vbm, self.cbm = self.set_vbm_cbm(vbm, cbm)
        self.mark_fermi_level = mark_fermi_level
        self.mark_gap = mark_gap
        self.fill = fill
        self.color = color
        self.show_total = show_total
        self.main = main
        self._handles = []

    def draw(self):
        xlabel, ylabel = self.get_xy_labels()
        x, y = self.x, self.y
        if self.broadening > 0.0:
            y = smear_dos(x, y, width=self.broadening)
        if self.flip:
            x, y = self.switch([x, y])
            xlabel, ylabel = self.switch((xlabel, ylabel))
            self.ax.set_xticks([])
            ylocs = ticker.MultipleLocator(base=0.5)
            self.ax.yaxis.set_major_locator(ylocs)
        else:
            self.ax.set_yticks([])
            xlocs = ticker.MultipleLocator(base=0.5)
            self.ax.xaxis.set_major_locator(xlocs)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if self.fill == "gradient":
            self.ax = gradient_fill(x, y, self.ax, self.color, flip=self.flip)
        if self.mark_fermi_level not in ["None", "none", None, False] and self.main:
            self.ax = self.show_fermi_level()
        if self.show_total not in ["None", None, False, []] and self.main:
            tx, ty = self.get_total_dos()
            ls = (0, (1, 1))
            self.handles.append(
                Line2D(
                    [0],
                    [0],
                    color=mutedblack,
                    label="total",
                    linestyle=(0, (1, 1)),
                    lw=1.0,
                )
            )
            if self.flip:
                tx, ty = self.switch([tx, ty])
            self.ax.plot(tx, ty, color=mutedblack, linestyle=ls)
        xlimits, ylimits = self.get_xy_limits()
        ylimits = ylimits[0] * 1.05, ylimits[1] * 1.05
        if self.flip:
            xlimits, ylimits = self.switch((xlimits, ylimits))
        self.ax.set_xlim(xlimits)
        self.ax.set_ylim(ylimits)
        latex_symbol = self.con.get_latex_symbol()
        self._handles.append(
            Line2D([0], [0], color=self.color, label=latex_symbol, lw=1.0)
        )
        self.xy = (x, y)
        return self.ax

    def switch(self, pair):
        a = pair[0]
        b = pair[1]
        return (b, a)

    def set_window(self, window):
        if (type(window) == float) or (type(window) == int):
            w = -window, window
            if self.ref in ["work function", "user-specified"]:
                w = -window - self.shift, window - self.shift
        elif len(window) == 2:
            w = window[0], window[1]
        else:
            raise Exception("Window not recognized.")
        return w

    def get_xy_labels(self):
        if self.ref in ["fermi level", "VBM", "middle"]:
            xlabel = r"E - E$_{\mathrm{F}}$ [eV]"
        elif self.ref == "work function":
            xlabel = r"E - E$_{vacuum}$ [eV]"
        else:
            xlabel = r"E [eV]"
        ylabel = r"DOS [1/eV]"
        return (xlabel, ylabel)

    def get_xy_limits(self):
        # I'm always setting the limits relative to the total dos
        window = self.window
        x, y = self.get_total_dos()

        lower_elimit, upper_elimit = window
        # assert that DOS either starts at 0 or a negative peak for spin
        ly = np.min(y[(x >= lower_elimit) & (x <= upper_elimit)])
        ly = min([0, ly])
        uy = np.max(y[(x >= lower_elimit) & (x <= upper_elimit)])
        uy = max([0, uy])
        return [(lower_elimit, upper_elimit), (ly, uy)]

    def show_fermi_level(self):
        ref = self.ref
        value = self.shift
        color = self.mark_fermi_level
        if ref == "work function":
            mark = -value
        else:
            mark = 0.00
        if self.flip:
            self.ax.axhline(
                y=mark, color=color, alpha=fermi_alpha, linestyle="--",
            )
        else:
            self.ax.axvline(
                x=mark, color=color, alpha=fermi_alpha, linestyle="--",
            )
        return self.ax

    def get_total_dos(self):
        x = self.x
        s = 1 if self.spin == 0 else -1
        y = self.show_total.values[:, self.spin, 0] * s
        if self.broadening > 0.0:
            y = smear_dos(x, y, self.broadening)
        return x, y

    def set_vbm_cbm(self, vbm, cbm):
        if self.ref not in ["work function", "user-specified"]:
            vbm -= self.fermi_level
            cbm -= self.fermi_level
        else:
            vbm -= self.shift
            cbm -= self.shift
        return vbm, cbm

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
    def __init__(self, energies, contributions, type):
        self.energies = energies
        self.contributions = contributions
        self.type = type

    def __repr__(self) -> str:
        return "{}(type={})".format(self.__class__.__name__, self.type)

    def get_total_dos(self):
        tdos = sum([k for k in self.contributions])
        return tdos

    def get_species_contribution(self, symbol):
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
        assert (
            self.type == "atom"
        ), "This spectrum type does not support atom contributions."
        con = self.contributions[index]
        return con


class Contribution:
    def __init__(self, symbol, values) -> None:
        self._symbol = symbol
        self.values = values

    def __repr__(self) -> str:
        return "{}({})".format(self.__class__.__name__, self.symbol)

    def __add__(self, other) -> "Contribution":
        d = self.values + other.values
        s = (
            Formula.from_list([self.symbol, other.symbol])
            .format("metal")
            .format("reduce")
        )
        return Contribution(s, d)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    @property
    def symbol(self):
        return self._symbol

    def set_symbol(self, symbol):
        assert type(symbol) == str, "Symbol must be a string."
        try:
            s = string2symbols(symbol)
        except Exception as expt:
            raise Exception("String could not be interpreted as atomic symbols.")
        assert all(
            k in chemical_symbols for k in s
        ), "Symbol is not an element from the PSE."
        s = Formula.from_list(s).format("metal").format("reduce")
        self._symbol = s

    def get_latex_symbol(self):
        s = self.symbol
        s = Formula(s)
        return s.format("latex")
