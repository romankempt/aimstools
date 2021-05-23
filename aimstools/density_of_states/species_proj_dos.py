from aimstools.density_of_states.total_dos import TotalDOS
from aimstools.misc import *
from aimstools.density_of_states.base import DOSBaseClass
from aimstools.density_of_states.utilities import DOSSpectrum, DOSPlot

import numpy as np

import re

from ase.data.colors import jmol_colors
from ase.data import chemical_symbols, atomic_masses
from ase.symbols import symbols2numbers, string2symbols

from matplotlib.colors import is_color_like


class SpeciesProjectedDOS(TotalDOS, DOSBaseClass):
    def __init__(self, outputfile, soc=False) -> None:
        DOSBaseClass.__init__(self, outputfile)
        assert any(
            x in ["species-projected dos", "species-projected dos tetrahedron"]
            for x in self.tasks
        ), "Species-projected DOS was not specified as task in control.in ."
        self.soc = soc
        self.spin = "none" if self.control["spin"] != "collinear" else "collinear"
        self._spectrum = self.set_spectrum(None)

    def __repr__(self):
        return "{}(outputfile={}, spin_orbit_coupling={})".format(
            self.__class__.__name__, repr(self.outputfile), self.soc
        )

    def _read_dosfiles(self):
        if self.spin == "none":
            dosfiles = self.get_dos_files(soc=self.soc, spin="none").species_proj_dos
            dosfiles = list(zip(dosfiles, dosfiles))
        if self.spin == "collinear":
            dosfiles_dn = self.get_dos_files(soc=self.soc, spin="dn").species_proj_dos
            dosfiles_up = self.get_dos_files(soc=self.soc, spin="up").species_proj_dos
            dosfiles = list(zip(dosfiles_dn, dosfiles_up))
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
            dos_per_species.append((symbol, contributions))
        self._dos = (energies, dos_per_species)

    def set_spectrum(self, reference=None):
        if self.dos == None:
            self._read_dosfiles()
        energies, dos_per_species = self.dos
        self.set_energy_reference(reference, self.soc)
        fermi_level = self.fermi_level.soc if self.soc else self.fermi_level.scalar
        reference, shift = self.energy_reference
        band_extrema = self.band_extrema[:2] if not self.soc else self.band_extrema[2:]
        atoms = self.structure.atoms
        self._spectrum = DOSSpectrum(
            atoms=atoms,
            energies=energies,
            contributions=dos_per_species,
            type="species",
            fermi_level=fermi_level,
            reference=reference,
            band_extrema=band_extrema,
            shift=shift,
        )

    @property
    def spectrum(self):
        ":class:`aimstools.density_of_states.utilities.DOSSpectrum`."
        if self._spectrum == None:
            self.set_spectrum(None)
        return self._spectrum

    def _process_contributions(self, contributions):
        """Helper function to format list of contributions."""

        momenta = ["tot", "s", "p", "d", "f", "g", "h"]
        if isinstance(contributions, (str, int)):
            contributions = [contributions]
        elif isinstance(contributions, (list, tuple)):
            assert len(contributions) > 0, "Contributions cannot be empty list."
            if isinstance(contributions, (list, tuple)) and len(contributions) == 2:
                if isinstance(contributions[0], (str, int)) and isinstance(
                    contributions[1], (str)
                ):
                    # corner case if only (symbol, l) is specified
                    if contributions[1] in momenta:
                        contributions = [contributions]

        if isinstance(contributions, (list, tuple)):
            for i, j in enumerate(contributions):
                if isinstance(j, (list, tuple)):
                    assert len(j) == 2, f"Too many entries specified for component {j}."
                    assert isinstance(
                        j[0], (str, int, tuple)
                    ), f"Contribution identifier of {j} must be string, tuple or integer index."
                    assert isinstance(
                        j[1], (str,)
                    ), f"Angular momentum of {j} identifier must be string."
                elif isinstance(j, (str, int)):
                    assert (
                        j not in momenta
                    ), f"Angular momentum specified at position {i} without contribution identifier."
                    contributions[i] = (j, "tot")
        else:
            raise Exception("Contributions not recognized.")

        new_cons = []
        for con in contributions:
            i, l = con
            if isinstance(i, (int,)):
                assert i in range(
                    len(self.structure)
                ), f"Index {i} is out of bounds for length of atoms."
                new_cons.append(self.spectrum.get_atom_contribution(i, l))
            if isinstance(i, (tuple,)):
                indices = [j for j in i]
                for j in indices:
                    assert j in range(
                        len(self.structure)
                    ), f"Index {i} is out of bounds for length of atoms."
                new_cons.append(
                    sum([self.spectrum.get_atom_contribution(k, l) for k in indices])
                )
            if isinstance(i, (str,)):
                if i == "all":
                    i = self.structure.get_chemical_formula()
                try:
                    s = string2symbols(i)
                except Exception as expt:
                    raise Exception(
                        "String could not be interpreted as atomic symbols."
                    )
                new_cons.append(self.spectrum.get_group_contribution(s, l=l))
        return new_cons

    def plot_contributions(
        self,
        axes: "matplotlib.axes.Axes" = None,
        contributions: list = [],
        colors: list = [],
        labels: list = [],
        **kwargs,
    ):
        """Main function to handle plotting of projected DOS.

        Contributions should be a list of identifiers that can be interpreted by :class:`~aimstools.density_of_states.utilities.DOSSpectrum`.
        They are converted to a list of :class:`~aimstools.density_of_states.utilities.DOSContribution`.
        
        The following combinations of contribution formats are accepted:
            - integers containing atom indices (e.g., [0,1,2,3])
            - species symbols (e.g., ["Mo", "S"])
            - symbol groups (e.g., ["MoS", "CH"])
            - tuples specifying an identifier and an angular momentum (e.g., [("all", "s")])
            - tuples containing a tuple of atom indices and an angular momentum (e.g., [((0,1,2), "tot")])

        Example:
            >>> from aimstools.density_of_states import AtomProjectedDOS as APD
            >>> dos = APD("path/to/dir")
            >>> dos.plot_contributions(contributions=["F", "CH"], labels=["foo", "bar"], colors=["green", "blue"], mode="scatter")

        Args:            
            contributions (list, optional): List of contribution identifiers. Defaults to [].
            colors (list, optional): List of colors. Defaults to [].
            labels (list, optional): List of labels. Defaults to [].
            show_legend (bool, optional): Show legend for labels and colors. Defaults to True.
            legend_linewidth (float, optional): Legend handle linewidth. Defaults to 1.5.
            legend_frameon (bool, optional): Show legend frame. Defaults to True.
            legend_fancybox (bool, optional): Enable bevelled box. Defaults to True.
            legend_borderpad (float, optional): Pad for legend borders. Defaults to 0.4.
            legend_loc (string, optional): Legend location. Defaults to "upper right".
            legend_handlelength (float): Legend handlelength, defaults to 0.4.
            axes (matplotlib.axes.Axes): Axes to draw on, defaults to None.
            figsize (tuple): Figure size in inches. Defaults to (5,5).
            filename (str): Saves figure to file. Defaults to None.
            spin (int): Spin channel, can be "up", "dn", 0 or 1. Defaults to 0.       
            reference (str): Energy reference for plotting, e.g., "VBM", "middle", "fermi level". Defaults to None.
            show_fermi_level (bool): Show Fermi level. Defaults to True.
            fermi_level_color (str): Color of Fermi level line. Defaults to fermi_color.
            fermi_level_alpha (float): Alpha channel of Fermi level line. Defaults to 1.0.
            fermi_level_linestyle (str): Line style of Fermi level line. Defaults to "--".
            fermi_level_linewidth (float): Line width of Fermi level line. Defaults to mpllinewidth.
            show_grid_lines (bool): Show grid lines for axes ticks. Defaults to False.
            grid_lines_axes (str): Show grid lines for given axes. Defaults to "x".
            grid_linestyle (tuple): Grid lines linestyle. Defaults to (0, (1, 1)).
            grid_linewidth (float): Width of grid lines. Defaults to 1.0.
            grid_linecolor (str): Grid lines color. Defaults to mutedblack.
            window (tuple): Window on energy axis, can be float or tuple of two floats in eV. Defaults to 3 eV.
            energy_tick_locator (float): Places ticks on energy axis on regular intervals. Defaults to 0.5 eV.     
            dos_tick_locator (float): Places ticks on dos axis on regular intervals. Defaults to 1 state / eV.       
            broadening (float): Smears DOS by finite Lorentzian width. Defaults to 0.00 eV.
       
        Returns:
            axes: Axes object.
        """

        kwargs = self._process_kwargs(kwargs)
        reference = kwargs.pop("reference", None)
        spectrum = self.get_spectrum(reference=reference)
        contributions = self._process_contributions(contributions)

        if isinstance(colors, (list, np.array)):
            if len(colors) == 0:
                cmap = plt.cm.get_cmap("tab10")
                colors = [cmap(c) for c in np.linspace(0, 1, len(contributions))]
            assert all(is_color_like(j) for j in colors), "Colors must be color-like."
        elif is_color_like(colors):
            colors = [colors]
        else:
            raise Exception("Colors not recognized.")

        if isinstance(labels, (list, tuple)):
            if len(labels) == 0:
                labels = [c.get_latex_symbol() for c in contributions]
            else:
                assert all(
                    isinstance(j, (str,)) for j in labels
                ), "Labels must be strings."
        elif isinstance(labels, str):
            labels = [labels]
        else:
            raise Exception("Labels not recognized.")

        assert len(contributions) == len(
            colors
        ), "Length of contributions and colors does not match."
        assert len(contributions) == len(
            labels
        ), "Length of contributions and labels does not match."

        with AxesContext(ax=axes, **kwargs) as axes:
            dosplot = DOSPlot(
                ax=axes,
                spectrum=spectrum,
                contributions=contributions,
                labels=labels,
                colors=colors,
                **kwargs,
            )
            dosplot.draw()

        return axes

    def plot_all_species(
        self, axes=None, l="tot", colors=[], show_legend=True, **kwargs
    ):
        """Utility function to all show species contributions.

        Args:
            l (str, optional): Angular momentum. Defaults to "tot".
        """
        kwargs = self._process_kwargs(kwargs)
        species = list(
            dict.fromkeys(string2symbols(self.structure.get_chemical_formula()))
        )
        contributions = [(j, l) for j in species]
        labels = species

        if len(colors) == 0:
            colors = [jmol_colors[symbols2numbers(n)][0] for n in labels]

        assert len(labels) == len(
            colors
        ), "Number of symbols does not match number of colors."

        self.plot_contributions(
            axes=axes,
            contributions=contributions,
            colors=colors,
            labels=labels,
            show_legend=show_legend,
            **kwargs,
        )
        return axes

    def plot_all_angular_momenta(
        self, max_l="f", axes=None, colors=[], show_colorbar=True, **kwargs
    ):
        """Utility function to show contributions of angular momenta (e.g., "s", "p", "d" orbitals).

        Args:
            max_l (str, optional): Maximum angular momentum to show. Defaults to "f".
        """
        kwargs = self._process_kwargs(kwargs)

        momenta = ("s", "p", "d", "f", "g", "h")
        momenta = dict(zip(momenta, range(len(momenta))))
        momenta = {k: v for k, v in momenta.items() if v <= momenta[max_l]}

        contributions = [("all", l) for l in momenta.keys()]

        if colors == []:
            cmap = plt.cm.get_cmap("tab10")
            colors = [cmap(c) for c in np.linspace(0, 1, 6)]
            colors = colors[: len(momenta)]

        self.plot_contributions(
            axes=axes,
            contributions=contributions,
            colors=colors,
            labels=list(momenta.keys()),
            show_colorbar=show_colorbar,
            **kwargs,
        )
        return axes
