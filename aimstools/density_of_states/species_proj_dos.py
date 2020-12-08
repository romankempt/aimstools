from aimstools.misc import *
from aimstools.density_of_states.base import DOSBaseClass
from aimstools.density_of_states.utilities import DOSPlot, Contribution, DOSSpectrum

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm
import numpy as np

import re

from collections import namedtuple

from ase.data.colors import jmol_colors
from ase.formula import Formula
from ase.symbols import Symbols, string2symbols, symbols2numbers
from ase.data import chemical_symbols, atomic_masses


class SpeciesProjectedDOSMethods:
    def _process_kwargs(self, **kwargs):
        kwargs = kwargs.copy()

        axargs = {}
        axargs["figsize"] = kwargs.pop("figsize", (3, 6))
        axargs["filename"] = kwargs.pop("filename", None)
        axargs["title"] = kwargs.pop("title", None)

        d = {}
        spin = kwargs.pop("spin", None)
        reference = kwargs.pop("reference", None)

        d["flip"] = kwargs.pop("flip", True)
        d["window"] = kwargs.pop("window", 3)
        d["mark_fermi_level"] = kwargs.pop("mark_fermi_level", fermi_color)
        d["broadening"] = kwargs.pop("broadening", 0.0)
        d["fill"] = kwargs.pop("fill", "gradient")
        d["show_total"] = kwargs.pop("show_total", True)
        show_total = kwargs.pop("show_total", True)

        if show_total:
            d["show_total"] = self.spectrum.get_total_dos()
        self.set_energy_reference(reference, self.soc)
        ref, shift = self.energy_reference
        fermi_level = self.fermi_level.soc if self.soc else self.fermi_level.scalar
        be = self.band_extrema
        vbm = be.vbm_soc if self.soc else be.vbm_scalar
        cbm = be.cbm_soc if self.soc else be.cbm_scalar

        spin = self.spin2index(spin)
        if self.soc and spin == 1:
            raise Exception(
                "Spin channels are ill-defined for SOC calculations. A second spin channel does not exist."
            )

        d["spin"] = spin
        d["vbm"] = vbm
        d["cbm"] = cbm
        d["ref"] = ref
        d["shift"] = shift
        d["fermi_level"] = fermi_level
        return axargs, kwargs, d

    def plot_one_species(
        self,
        symbol,
        l="tot",
        axes=None,
        color=None,
        main=True,
        **kwargs,
    ):
        axargs, kwargs, dosargs = self._process_kwargs(**kwargs)
        assert (
            symbol in self.structure.symbols
        ), "The species {} was not found in the structure.".format(symbol)
        x = self.spectrum.energies
        con = self.spectrum.get_species_contribution(symbol)

        number = symbols2numbers(symbol)[0]
        color = jmol_colors[number] if type(color) == type(None) else color

        with AxesContext(ax=axes, main=main, **axargs) as axes:
            dosplot = DOSPlot(x=x, con=con, l="tot", color=color, main=main, **dosargs)
            axes = dosplot.draw()
            x, y = dosplot.xy
            axes.plot(x, y, color=color, **kwargs)
            axes.legend(
                handles=dosplot.handles,
                frameon=True,
                loc="center right",
                fancybox=False,
            )
        return axes

    def plot_all_species(
        self,
        symbols=[],
        l="tot",
        axes=None,
        colors=[],
        main=True,
        **kwargs,
    ):
        axargs, kwargs, dosargs = self._process_kwargs(**kwargs)

        if symbols in ["all", "None", None, "All", []]:
            symbols = set([k.symbol for k in self.structure])
        else:
            symbols = set(symbols)

        if len(colors) == 0:
            colors = [jmol_colors[symbols2numbers(n)][0] for n in symbols]

        assert len(symbols) == len(
            colors
        ), "Number of symbols does not match number of colors."

        masses = [atomic_masses[symbols2numbers(m)] for m in symbols]
        scm = tuple(
            sorted(zip(symbols, colors, masses), key=lambda x: x[2], reverse=True)
        )
        handles = []
        with AxesContext(ax=axes, main=main, **axargs) as axes:
            for i, (s, c, _) in enumerate(scm):
                m = main if i == 0 else False
                x = self.spectrum.energies
                con = self.spectrum.get_species_contribution(s)
                dosplot = DOSPlot(x=x, con=con, l=l, main=m, color=c, **dosargs)
                axes = dosplot.draw()
                x, y = dosplot.xy
                axes.plot(x, y, color=c, **kwargs)
                handles += dosplot.handles
            axes.legend(
                handles=handles, frameon=True, loc="center right", fancybox=False
            )
        return axes

    def plot_all_angular_momenta(
        self,
        symbols=[],
        max_l="f",
        axes=None,
        colors=[],
        main=True,
        **kwargs,
    ):
        axargs, kwargs, dosargs = self._process_kwargs(**kwargs)
        momenta = ("s", "p", "d", "f", "g", "h")
        momenta = dict(zip(momenta, range(len(momenta))))
        momenta = {k: v for k, v in momenta.items() if v <= momenta[max_l]}
        if symbols in ["all", "None", None, "All", []]:
            symbols = set([k.symbol for k in self.structure])
        else:
            symbols = set(symbols)

        if colors == []:
            cmap = matplotlib.cm.get_cmap("tab10")
            colors = [cmap(k) for k in np.arange(0, 7, 1)]

        handles = []
        with AxesContext(ax=axes, main=main, **axargs) as axes:
            en = self.spectrum.energies
            con = sum([self.spectrum.get_species_contribution(k) for k in symbols])
            for i, (label, l) in enumerate(momenta.items()):
                m = main if i == 0 else False
                dosplot = DOSPlot(
                    x=en, con=con, l=l, color=colors[i], main=m, **dosargs
                )
                axes = dosplot.draw()
                x, y = dosplot.xy
                axes.plot(x, y, color=colors[i], **kwargs)
                handles.append(Line2D([0], [0], color=colors[i], label=label, lw=1.0))
            if dosargs["show_total"] not in [False, "None", "none", None, []]:
                handles.insert(
                    0,
                    Line2D(
                        [0],
                        [0],
                        color=mutedblack,
                        label="total",
                        linestyle=(0, (1, 1)),
                        lw=1.0,
                    ),
                )
            axes.legend(
                handles=handles, frameon=True, loc="center right", fancybox=False
            )
        return axes

    def plot_custom_contributions(
        self,
        list_of_contributions,
        colors=[],
        labels=[],
        l="tot",
        axes=None,
        main=True,
        **kwargs,
    ):
        axargs, kwargs, dosargs = self._process_kwargs(**kwargs)
        if type(list_of_contributions) not in [list, tuple]:
            list_of_contributions = [list_of_contributions]
        if type(colors) not in [list, tuple]:
            colors = [colors]
        if type(labels) not in [list, tuple]:
            labels = [labels]
        if colors in [[], None]:
            cmap = matplotlib.cm.get_cmap("tab10")
            mc = len(list_of_contributions)
            colors = [cmap(k) for k in np.arange(0, mc, 1)]

        assert len(list_of_contributions) == len(
            colors
        ), "Number of contributions does not match number of colors."

        if labels in [None, []]:
            labels = [s.get_latex_symbol() for s in list_of_contributions]

        assert len(list_of_contributions) == len(
            labels
        ), "Number of contributions does not match number of labels."

        handles = [
            Line2D([0], [0], label=l, color=c, lw=1.0) for l, c in zip(labels, colors)
        ]
        with AxesContext(ax=axes, main=main, **axargs) as axes:
            ev = self.spectrum.energies
            for i, con in enumerate(list_of_contributions):
                m = main if i == 0 else False
                dosplot = DOSPlot(
                    x=ev, con=con, l=l, color=colors[i], main=m, **dosargs
                )
                axes = dosplot.draw()
                x, y = dosplot.xy
                axes.plot(x, y, color=colors[i], **kwargs)
            if dosargs["show_total"] not in [False, "None", "none", None, []]:
                handles.insert(
                    0,
                    Line2D(
                        [0],
                        [0],
                        color=mutedblack,
                        label="total",
                        linestyle=(0, (1, 1)),
                        lw=1.0,
                    ),
                )
            axes.legend(
                handles=handles, frameon=True, loc="center right", fancybox=False
            )
        return axes


class SpeciesProjectedDOS(DOSBaseClass, SpeciesProjectedDOSMethods):
    def __init__(self, outputfile, soc=False) -> None:
        super().__init__(outputfile)
        assert any(
            x in ["species-projected dos", "species-projected dos tetrahedron"]
            for x in self.tasks
        ), "Species-projected DOS was not specified as task in control.in ."
        self.soc = soc
        self.spin = "none" if self.control["spin"] != "collinear" else "collinear"
        if self.spin == "none":
            dosfiles = self.get_dos_files(soc=soc, spin="none").species_proj_dos
            spectrum = self.read_dosfiles(list(zip(dosfiles, dosfiles)))
        if self.spin == "collinear":
            dosfiles_dn = self.get_dos_files(soc=soc, spin="dn").species_proj_dos
            dosfiles_up = self.get_dos_files(soc=soc, spin="up").species_proj_dos
            spectrum = self.read_dosfiles(list(zip(dosfiles_dn, dosfiles_up)))
        assert (
            len(spectrum.contributions) > 0
        ), "Couldn't read dosfiles, something must have gone wrong."
        self.spectrum = spectrum

    def __repr__(self):
        return "{}(outputfile={}, spin_orbit_coupling={})".format(
            self.__class__.__name__, repr(self.outputfile), self.soc
        )

    def read_dosfiles(self, dosfiles):
        dos_per_species = []
        energies = []
        nspins = 2 if self.spin == "collinear" else 1
        for symbol in set(self.structure.symbols):
            pattern = re.compile(symbol + r"\_l")
            energies = []
            contributions = []
            for s in range(nspins):
                atom_file = [k[s] for k in dosfiles if re.search(pattern, str(k[s]))]
                assert (
                    len(atom_file) == 1
                ), "Multiple species-projected dos files found for same species. Something must have gone wrong. Found: {}".format(
                    atom_file
                )
                array = np.loadtxt(atom_file[0], dtype=float, comments="#")
                ev, co = array[:, 0], array[:, 1:]
                # This ensures that all arrays have shape (nenergies, 7)
                nrows, ncols = co.shape
                array = np.zeros((nrows, 7))
                array[:, :ncols] = co
                energies.append(ev)
                contributions.append(array)
            energies = np.stack(energies, axis=1)
            contributions = np.stack(contributions, axis=1)
            con = Contribution(symbol, contributions)
            dos_per_species.append(con)
        spectrum = DOSSpectrum(energies, dos_per_species, "species")
        return spectrum
