from aimstools.structuretools import structure
from aimstools.misc import *
from aimstools.bandstructures.utilities import (
    BandSpectrum,
    BandStructurePlot,
    MullikenBandStructurePlot,
)
from aimstools.bandstructures.bandstructure import BandStructureBaseClass

import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like

from ase.dft.kpoints import parse_path_string
from ase.symbols import string2symbols, symbols2numbers
from ase.data import chemical_symbols, atomic_masses
from ase.data.colors import jmol_colors
from ase.formula import Formula

from collections import namedtuple
import numpy as np
import time


class MullikenSpectrum(BandSpectrum):
    """Container class for eigenvalue spectrum and mulliken contributions.

    Attributes:
        contributions (MullikenContribution): :class:`~aimstools.bandstructures.mulliken_bandstructure.MullikenContribution`.

    """

    def __init__(
        self,
        atoms: "ase.atoms.Atoms" = None,
        kpoints: "numpy.ndarray" = None,
        kpoint_axis: "numpy.ndarray" = None,
        eigenvalues: "numpy.ndarray" = None,
        occupations: "numpy.ndarray" = None,
        contributions: "numpy.ndarray" = None,
        label_coords: list = None,
        kpoint_labels: list = None,
        jumps: list = None,
        fermi_level: float = None,
        reference: str = None,
        shift: float = None,
        bandpath: str = None,
    ) -> None:
        super().__init__(
            atoms=atoms,
            kpoints=kpoints,
            kpoint_axis=kpoint_axis,
            eigenvalues=eigenvalues,
            occupations=occupations,
            label_coords=label_coords,
            kpoint_labels=kpoint_labels,
            jumps=jumps,
            fermi_level=fermi_level,
            reference=reference,
            shift=shift,
            bandpath=bandpath,
        )
        self._contributions = contributions

    @property
    def contributions(self):
        return self._contributions

    def get_atom_contribution(self, index, l="tot"):
        """Returns :class:`~aimstools.bandstructures.mulliken_bandstructure.MullikenContribution` of given atom index.

        Args:
            index (int): Index of atom.
            l (optional): Angular momentum. Defaults to "tot".
        """
        l = self._l2index(l)
        s = self.atoms[index].symbol
        con = self.contributions[index, :, :, :, l].copy()
        return MullikenContribution(s, con, l)

    def get_symbol(self, symbol):
        """Formats given symbol."""
        assert type(symbol) == str, "Symbol must be a string."
        try:
            s = string2symbols(symbol)
        except Exception as expt:
            raise Exception("String could not be interpreted as atomic symbols.")
        assert all(
            k in chemical_symbols for k in s
        ), "Symbol is not an element from the PSE."
        s = Formula.from_list(s).format("metal").format("reduce")
        return s

    def get_species_contribution(self, symbol, l="tot"):
        """Returns :class:`~aimstools.bandstructures.mulliken_bandstructure.MullikenContribution` of given species symbol.

        Args:
            symbol (str): Species symbol, e.g., "C".
            l (optional): Angular momentum. Defaults to "tot".
        """
        symbol = self.get_symbol(symbol)
        l = self._l2index(l)
        assert symbol in self.atoms.symbols, "Symbol {} not part of atoms.".format(
            symbol
        )
        indices = [k for k, j in enumerate(self.atoms) if j.symbol == symbol]
        cons = self.contributions[indices, ...]
        cons = np.sum(cons[:, :, :, :, l], axis=0)
        return MullikenContribution(symbol, cons, l)

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
            raise Exception("Implemented mlk bandstructures only till h-orbitals.")

    def __repr__(self):
        return "{}(bandpath={}, reference={}), band_gap={}, contribution_species={}".format(
            self.__class__.__name__,
            self.bandpath,
            self.reference,
            self.bandgap,
            list(set(self.atoms.symbols)),
        )


class MullikenContribution:
    """Container class to hold mulliken-projected atomic contributions to the band structure.

    MullikenContribution supports addition and substraction.

    Attributes:
        contribution (ndarray): (natoms, nkpoints, nspins, nstates, 5) array with the last five axes being the angular momenta (tot, s, p, d, f).
    """

    def __init__(self, symbol, contribution, l) -> None:
        self._symbol = symbol
        self.contribution = contribution
        self._l = self._index2l(l)

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

    def __repr__(self) -> str:
        return "MullikenContribution({}, {})".format(self.symbol, self.l)

    def __add__(self, other) -> "MullikenContribution":
        l = "".join(set([self.l, other.l]))
        d = self.contribution + other.contribution
        s1 = string2symbols(self.symbol)
        s2 = string2symbols(other.symbol)
        s = Formula.from_list(s1 + s2).format("reduce")
        return MullikenContribution(s, d, l)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self, other):
        l = "".join(set([self.l, other.l]))
        d = self.contribution - other.contribution
        s = Formula.from_list([self.symbol, other.symbol]).format("reduce")
        s = "$\Delta$" + s
        return MullikenContribution(s, d, l)

    @property
    def symbol(self):
        """Formatted symbol of the contribution."""
        return self._symbol

    @property
    def con(self):
        """Short-hand for self.contribution"""
        return self.contribution

    @property
    def l(self):
        """Angular momentum."""
        return self._l

    def get_latex_symbol(self):
        """Returns latex-formatted symbol string."""
        s = self.symbol
        s = Formula(s)
        return s.format("latex")


class MullikenBandStructure(BandStructureBaseClass):
    """Mulliken-projected band structure object.

    A mulliken-projected band structure shows the momentum-resolved Mulliken contribution of each atom to the energy.

    Attributes:
        spectrum: :class:`~aimstools.bandstructures.mulliken_bandstructure.MullikenSpectrum`.        
    """

    def __init__(self, outputfile, soc=False) -> None:
        super().__init__(outputfile)
        self.soc = soc
        self.band_sections = self.band_sections.mlk
        self._set_bandpath_from_sections()
        self.task = "mulliken-projected band structure"
        self.spin = "none" if self.control["spin"] != "collinear" else "collinear"
        if self.soc == False and self.spin == "collinear":
            raise Exception(
                "Spin files without SOC have yet another structure, which I have not implemented yet..."
            )
        if self.control["include_spin_orbit"] and not soc:
            raise Exception(
                "With include_spin_orbit, the scalar-relativistic mulliken band files are not written out."
            )
        bandfiles = self.get_bandfiles(spin=self.spin, soc=soc)
        self.bandfiles = bandfiles.mulliken
        self.bands = self._read_mlk_bandfiles(spin=self.spin)
        self._spectrum = self.set_spectrum(None, None)

    def __repr__(self):
        return "{}(outputfile={}, spin_orbit_coupling={})".format(
            self.__class__.__name__, repr(self.outputfile), self.soc
        )

    def _read_mlk_bandfiles(self, spin="none"):
        # Turns out that any attemps to use regex (np.fromregex, re.finditer ...) are much slower in this case than just pure string matching.
        atoms = self.structure.copy()
        natoms = len(atoms)
        if spin == "none" and self.soc == False:
            nspins = 1
        else:
            nspins = 2
        bands = {}
        b = namedtuple("band", ["kpoints", "data"])
        logger.info("Reading in mulliken bandfiles in serial ...")
        logger.debug(
            "Note: I'm forcing all l-contributions below zero to be zero, see discussion with Volker Blum."
        )
        for section, bandfile in zip(self.band_sections, self.bandfiles):
            start = time.time()
            with open(bandfile, "r") as file:
                lines = file.readlines()
            kpoints = np.array(
                [
                    l.strip().split()[-4:-1]
                    for l in [k for k in lines if "k point number" in k]
                ],
                dtype=np.float32,
            )
            values = [
                k.strip().split()
                for k in lines
                if "k point number" not in k and "State" not in k and k != ""
            ]
            out = np.zeros((len(values), 12), dtype=np.float32)
            for i, k in enumerate(values):
                out[i][0 : len(k)] = k
            ncons = out.shape[1] - 4 - nspins + 1  # dropping state, atom, spin indices
            nstates = int(np.max(out[:, 0]) - np.min(out[:, 0])) + 1
            nkpoints = len(kpoints)
            indices = [1, 2] + list(range(-ncons, 0, 1))
            out = out[:, indices]
            out[:, 2:] = np.where(out[:, 2:] < 0.00, 0.00, out[:, 2:])
            out[:, 2] = np.sum(out[:, 3:], axis=1)  # recalculating total contribution
            logger.debug(
                "Found: {:d} kpoints, {:d} states, {:d} spins, {:d} atoms, {:d} contributions.".format(
                    nkpoints, nstates, nspins, natoms, ncons
                )
            )
            out = out.reshape(nkpoints, nstates * natoms * nspins, ncons + 2)
            out = out.transpose(0, 2, 1)
            out = out.reshape(nkpoints, ncons + 2, nstates, natoms, nspins)
            out = out.transpose(
                3, 0, 4, 2, 1
            )  # (natoms, nkpoints, nspins, nstates, ncons)
            if self.soc:
                # Removing second spin channel for soc calculations.
                out[:, :, 0, :, 2:] += out[:, :, 1, :, 2:]
                out[:, :, 1, :, 2:] = 0.0
            pathsegment = (section.symbol1, section.symbol2)
            bands[pathsegment] = b(kpoints, out)
            end = time.time()
            logger.info(
                "\t ... processed {} in {:.2f} seconds.".format(
                    str(bandfile.parts[-1]), end - start
                )
            )
        return bands

    def set_spectrum(self, bandpath=None, reference=None):
        """ Sets :class:`~aimstools.bandstructures.mulliken_bandstructure.MullikenSpectrum` for a given bandpath.
        
        Bandpath should be ASE-formatted string, e.g., "GMKG,A", where the "," denotes jumps.
        """
        bands = self.bands
        atoms = self.structure.copy()
        start = time.time()
        if bandpath != None:
            self.set_bandpath(bandpath)
        bp = parse_path_string(self.bandpath.path)
        jumps = []
        kps = []
        occs = []
        kpoint_axis = []
        kpoint_labels = []
        label_coords = []
        spectrum = []
        icell_cv = 2 * np.pi * np.linalg.pinv(self.structure.cell).T
        contributions = []
        for segment in bp:
            kpoint_labels.append(segment[0])
            label_coords.append(label_coords[-1] if len(label_coords) > 0 else 0.0)
            for s1, s2 in zip(segment[:-1], segment[1:]):
                if (s1, s2) in bands.keys():
                    data = bands[(s1, s2)].data
                    kpoints = np.dot(bands[(s1, s2)].kpoints, icell_cv)
                    energies = data[0, :, :, :, 0]  # (nkpoints, nspins, nstates)
                    occ = data[0, :, :, :, 1]  # (nkpoints, nspins, nstates)
                    con = data[
                        :, :, :, :, 2:
                    ]  # (natoms, nkpoints, nspins, nstates, [tot, s, p, d, f])
                elif (s1, s2) in [(k, j) for j, k in bands.keys()]:
                    data = bands[(s2, s1)].data
                    kpoints = np.dot(bands[(s2, s1)].kpoints, icell_cv)[::-1]
                    energies = data[0, :, :, :, 0]  # (nkpoints, nspins, nstates)
                    energies = energies[::-1, :, :]
                    occ = data[0, :, :, :, 1]  # (nkpoints, nspins, nstates)
                    occ = occ[::-1, :, :]
                    con = data[
                        :, :, :, :, 2:
                    ]  # (natoms, nkpoints, nspins, nstates, [tot, s, p, d, f])
                    con = con[:, ::-1, ...]
                else:
                    raise Exception(
                        "Neither {}-{} nor {}-{} were found.".format(s1, s2, s2, s1)
                    )

                kstep = np.linalg.norm(kpoints[-1, :] - kpoints[0, :])
                kaxis = np.linspace(0, kstep, kpoints.shape[0]) + label_coords[-1]
                kstep += label_coords[-1]
                spectrum.append(energies)
                kps.append(kpoints)
                occs.append(occ)
                contributions.append(con)
                kpoint_axis.append(kaxis)
                label_coords.append(kstep)
                kpoint_labels.append(s2)
            jumps.append(label_coords[-1])
        jumps = jumps[:-1]
        spectrum = np.concatenate(spectrum, axis=0)
        kps = np.concatenate(kps, axis=0)
        kpoint_axis = np.concatenate(kpoint_axis, axis=0)
        occs = np.concatenate(occs, axis=0)
        cons = np.concatenate(contributions, axis=1)
        fermi_level = self.fermi_level.soc if self.soc else self.fermi_level.scalar
        self.set_energy_reference(reference, self.soc)
        reference, shift = self.energy_reference
        spec = MullikenSpectrum(
            atoms=atoms,
            kpoints=kps,
            kpoint_axis=kpoint_axis,
            eigenvalues=spectrum,
            occupations=occs,
            contributions=cons,
            label_coords=label_coords,
            kpoint_labels=kpoint_labels,
            jumps=jumps,
            fermi_level=fermi_level,
            reference=reference,
            shift=shift,
            bandpath=bp,
        )
        end = time.time()
        logger.info(
            "Creating spectrum from bands took {:.2f} seconds.".format(end - start)
        )
        self._spectrum = spec

    @property
    def spectrum(self):
        if self._spectrum == None:
            self._spectrum = self.set_spectrum(bandpath=None, reference=None)
        return self._spectrum

    def get_spectrum(self, bandpath=None, reference=None):
        self.set_spectrum(bandpath=bandpath, reference=reference)
        return self.spectrum

    def _process_kwargs(self, kwargs):
        kwargs = kwargs.copy()

        spin = kwargs.pop("spin", None)
        kwargs["show_bandstructure"] = kwargs.pop("show_bandstructure", False)

        deprecated = ["title", "mark_fermi_level", "mark_band_gap"]
        for dep in deprecated:
            if dep in kwargs.keys():
                kwargs.pop(dep)
                logger.warning(f"Keyword {dep} is deprecated.")

        kwargs["spin"] = self.spin2index(spin)

        return kwargs

    def plot(self, axes=None, **kwargs):
        """ Same as :func:`~aimstools.bandstructures.regular_bandstructure.RegularBandStructure.plot` """
        kwargs = self._process_kwargs(kwargs)
        kwargs["show_bandstructure"] = True
        bandpath = kwargs.pop("bandpath", None)
        reference = kwargs.pop("reference", None)
        spectrum = self.get_spectrum(bandpath=bandpath, reference=reference)

        with AxesContext(ax=axes, **kwargs) as axes:
            bs = BandStructurePlot(ax=axes, spectrum=spectrum, **kwargs)
            bs.draw()

        return axes

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
        """Main function to handle plotting of Mulliken spectra. Supports all keywords of :func:`~aimstools.bandstructures.regular_bandstructure.RegularBandStructure.plot`.

        Contributions should be a list of identifiers that can be interpreted by :class:`~aimstools.bandstructures.mulliken_bandstructure.MullikenSpectrum`.
        They are converted to a list of :class:`~aimstools.bandstructures.mulliken_bandstructure.MullikenContribution`.
        
        The following combinations of contribution formats are accepted:
            - integers containing atom indices (e.g., [0,1,2,3])
            - species symbols (e.g., ["Mo", "S"])
            - symbol groups (e.g., ["MoS", "CH"])
            - tuples specifying an identifier and an angular momentum (e.g., [("all", "s")])
            - tuples containing a tuple of atom indices and an angular momentum (e.g., [((0,1,2), "tot")])

        Example:
            >>> from aimstools.bandstructures import MullikenBandStructure as MBS
            >>> bs = MBS("path/to/dir")
            >>> bs.plot_contributions(contributions=["F", "CH"], labels=["foo", "bar"], colors=["green", "blue"], mode="scatter")

        Args:            
            contributions (list, optional): List of contribution identifiers. Defaults to [].
            colors (list, optional): List of colors. Defaults to [].
            labels (list, optional): List of labels. Defaults to [].
            mode (str, optional):  Plotting mode, can be "lines", "scatter", "majority" or "gradient". Defaults to "lines".
            capstyle (str, optional): Matplotlib linecollection capstyle. Defaults to "round".
            interpolate (bool, optional): Interpolate bands and contributions for smoother appearance. Defaults to False.
            interpolation_step (float, optional): Interpolation step for interpolate. Defaults to 0.01.
            scale_width (bool, optional): Scale point or line widths by contribution. Defaults to True.
            scale_width_factor (float, optional): Factor to scale line widths by contribution. Defaults to 2.
            show_legend (bool, optional): Show legend for labels and colors. Defaults to True.
            legend_linewidth (float, optional): Legend handle linewidth. Defaults to 1.5.
            legend_frameon (bool, optional): Show legend frame. Defaults to True.
            legend_fancybox (bool, optional): Enable bevelled box. Defaults to True.
            legend_borderpad (float, optional): Pad for legend bordrs. Defaults to 0.4.
            legend_loc (string, optional): Legend location. Defaults to "upper right".
            legend_handlelength (float): Legend handlelength, defaults to 0.4.
            show_colorbar (bool, optional): Show colorbar. Defaults to False.
       
        Returns:
            axes: Axes object.
        """

        kwargs = self._process_kwargs(kwargs)
        bandpath = kwargs.pop("bandpath", None)
        reference = kwargs.pop("reference", None)
        spectrum = self.get_spectrum(bandpath=bandpath, reference=reference)

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
            mbs = MullikenBandStructurePlot(
                ax=axes,
                spectrum=spectrum,
                contributions=contributions,
                labels=labels,
                colors=colors,
                **kwargs,
            )
            mbs.draw()

        return axes

    def plot_majority_contribution(
        self,
        contributions=[],
        axes=None,
        colors=[],
        labels=[],
        show_colorbar=True,
        **kwargs,
    ):
        """Utility function to show majority contributions of given list of contributions.

        A majority-projection only shows the largest contribution to each k-point and band.
        """
        kwargs = self._process_kwargs(kwargs)
        kwargs["mode"] = "majority"

        if contributions == []:
            contributions = list(set(self.structure.symbols))
            labels = list(set(self.structure.symbols))

        self.plot_contributions(
            axes=axes,
            contributions=contributions,
            colors=colors,
            labels=labels,
            show_colorbar=show_colorbar,
            **kwargs,
        )
        return axes

    def plot_all_species(
        self, axes=None, l="tot", colors=[], show_legend=True, **kwargs
    ):
        """Utility function to all show species contributions.

        Args:
            l (str, optional): Angular momentum. Defaults to "tot".
        """
        kwargs = self._process_kwargs(kwargs)
        contributions = [(j, l) for j in set(self.structure.symbols)]
        labels = set(self.structure.symbols)

        if len(colors) == 0:
            colors = [jmol_colors[symbols2numbers(n)][0] for n in labels]

        assert len(labels) == len(
            colors
        ), "Number of symbols does not match number of colors."

        masses = [atomic_masses[symbols2numbers(m)] for m in labels]
        scm = tuple(
            sorted(zip(labels, colors, masses), key=lambda x: x[2], reverse=True)
        )
        labels = [j[0] for j in scm]
        colors = [j[1] for j in scm]

        self.plot_contributions(
            axes=axes,
            contributions=contributions,
            colors=colors,
            labels=labels,
            show_legend=show_legend,
            **kwargs,
        )
        return axes

    def plot_difference_contribution(
        self,
        con1,
        con2,
        axes=None,
        colors=["blue", "red"],
        show_colorbar=True,
        **kwargs,
    ):
        """Utility function to show difference of two contributions with a color gradient."""
        kwargs = self._process_kwargs(kwargs)
        kwargs["mode"] = "gradient"
        contributions = [con1, con2]
        self.plot_contributions(
            axes=axes,
            contributions=contributions,
            colors=colors,
            show_colorbar=show_colorbar,
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
        kwargs["mode"] = "majority"
        kwargs["scale_linewidth"] = 2

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
