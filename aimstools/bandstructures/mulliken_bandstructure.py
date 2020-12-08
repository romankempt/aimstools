from aimstools.misc import *
from aimstools.bandstructures.utilities import (
    BandStructurePlot,
    MullikenBandStructurePlot,
)
from aimstools.bandstructures.bandstructure import BandStructureBaseClass

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.colors import Normalize

from ase.dft.kpoints import parse_path_string
from ase.symbols import string2symbols, symbols2numbers
from ase.data import chemical_symbols, atomic_masses
from ase.data.colors import jmol_colors
from ase.formula import Formula

from collections import namedtuple
import numpy as np
import time


class MullikenSpectrum:
    def __init__(
        self,
        atoms,
        kpoints,
        kpoint_axis,
        eigenvalues,
        occupations,
        contributions,
        label_coords,
        kpoint_labels,
        jumps,
    ) -> None:
        self.atoms = atoms
        self.kpoints = kpoints
        self.kpoint_axis = kpoint_axis
        self.eigenvalues = eigenvalues
        self.occupations = occupations
        self.contributions = contributions
        self.label_coords = label_coords
        self.kpoint_labels = kpoint_labels
        self.jumps = jumps

    def get_atom_contribution(self, index, l="tot"):
        l = self.l2index(l)
        s = self.atoms[index].symbol
        con = self.contributions[index, :, :, :, l].copy()
        return MullikenContribution(s, con, l)

    def get_symbol(self, symbol):
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
        symbol = self.get_symbol(symbol)
        l = self.l2index(l)
        assert symbol in self.atoms.symbols, "Symbol {} not part of atoms.".format(
            symbol
        )
        indices = [k for k, j in enumerate(self.atoms) if j.symbol == symbol]
        cons = self.contributions[indices, ...]
        cons = np.sum(cons[:, :, :, :, l], axis=0)
        return MullikenContribution(symbol, cons, l)

    def get_group_contribution(self, symbols, l="tot"):
        symbols = [self.get_symbol(s) for s in symbols]
        cons = sum([self.get_species_contribution(s, l) for s in symbols])
        return cons

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
            raise Exception("Implemented mlk bandstructures only till h-orbitals.")

    def __repr__(self):
        return "{}(atoms, kpoints, kpoint_axis, eigenvalues, occupations, contributions, label_coords, kpoint_labels, jumps)".format(
            self.__class__.__name__
        )


class MullikenContribution:
    def __init__(self, symbol, contribution, l) -> None:
        self._symbol = symbol
        self.contribution = contribution
        self._l = self.index2l(l)

    def index2l(self, l):
        if type(l) == int:
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
        elif type(l) == str:
            return l
        else:
            raise Exception("Angular momentum not recognized.")

    def __repr__(self) -> str:
        return "MullikenContribution({}, {})".format(self.symbol, self.l)

    def __add__(self, other) -> "MullikenContribution":
        l = "".join(set([self.l, other.l]))
        d = self.contribution + other.contribution
        s = Formula.from_list([self.symbol, other.symbol]).format("reduce")
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
        return self._symbol

    @property
    def con(self):
        return self.contribution

    @property
    def l(self):
        return self._l

    def get_latex_symbol(self):
        s = self.symbol
        s = Formula(s)
        return s.format("latex")


class MullikenBandStructure(BandStructureBaseClass):
    """Mulliken-projected band structure object.

    A fat band structure shows the momentum-resolved Mulliken contribution of each atom to the energy.
    """

    def __init__(self, outputfile, soc=False) -> None:
        super().__init__(outputfile)
        self.soc = soc
        self.band_sections = self.band_sections.mlk
        self._bandpath = self.set_bandpath()
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
        self.bands = self.read_mlk_bandfiles(spin=self.spin)
        self.spectrum = self.get_mlk_spectrum()

    def __repr__(self):
        return "{}(outputfile={}, spin_orbit_coupling={})".format(
            self.__class__.__name__, repr(self.outputfile), self.soc
        )

    def read_mlk_bandfiles(self, spin="none"):
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

    def get_mlk_spectrum(self, bandpath=None):
        bands = self.bands
        atoms = self.structure.copy()
        start = time.time()
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
                    energies = np.flip(energies, axis=0)
                    occ = data[0, :, :, :, 1]  # (nkpoints, nspins, nstates)
                    occ = np.flip(occ, axis=0)
                    con = data[
                        :, :, :, :, 2:
                    ]  # (natoms, nkpoints, nspins, nstates, [tot, s, p, d, f])
                    con = np.flip(con, axis=1)
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
        spec = MullikenSpectrum(
            atoms,
            kps,
            kpoint_axis,
            spectrum,
            occs,
            cons,
            label_coords,
            kpoint_labels,
            jumps,
        )
        end = time.time()
        logger.info(
            "Creating spectrum from bands took {:.2f} seconds.".format(end - start)
        )
        return spec

    def __write_contributions(self):
        arrs = []
        names = []
        for index, atom in self.atoms_to_plot.items():
            for segment in self.mlk_bandsegments.keys():
                name1 = "{}_{}_{}-{}_{}_fatband_kaxis.npy".format(
                    atom, index, segment[0], segment[1], self.nkpoints[segment]
                )
                names.append(name1)
                arrs.append(self.atom_contributions[index][segment][0])
                name2 = "{}_{}_{}-{}_{}_fatband_contribution.npy".format(
                    atom, index, segment[0], segment[1], self.nkpoints[segment]
                )
                names.append(name2)
                arrs.append(self.atom_contributions[index][segment][1])
        np.savez_compressed(
            self.path.joinpath("fatbands_atom_contributions.npz"),
            **dict(zip(names, arrs)),
        )

    def __read_contributions(self):
        atom_contributions = {}
        nkpoints = {}
        data = dict(np.load(str(self.path.joinpath("fatbands_atom_contributions.npz"))))
        for j, k in {a: b for a, b in data.items() if "kaxis" in a}.items():
            ident1 = j.split("_fatband_")
            for l, m in {a: b for a, b in data.items() if "contribution" in a}.items():
                ident2 = l.split("_fatband_")
                if ident1[0] == ident2[0]:
                    atom, index, segment, nk1 = j.split("_")[:4]
                    segment = (segment.split("-")[0], segment.split("-")[1])
                    kaxis = k
                    cons = m
            self.natoms = len(self.structure.atoms)
            self.nstates = int(cons.shape[1])
            self.ncons = int(cons.shape[2])
            nkpoints[segment] = int(nk1)
            index = int(index)
            if index not in atom_contributions.keys():
                atom_contributions[index] = {segment: (kaxis, cons)}
            else:
                atom_contributions[index].update({segment: (kaxis, cons)})
        self.nkpoints = nkpoints
        return atom_contributions

    def color_to_alpha_cmap(self, color):
        cmap = LinearSegmentedColormap.from_list("", ["white", color])
        my_cmap = cmap(np.arange(cmap.N))
        my_cmap[:, -1] = np.linspace(0, 1, cmap.N)  # this adds alpha
        my_cmap = ListedColormap(my_cmap)
        return my_cmap

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

        if spin != None and self.soc:
            logger.warning(
                "Spin channels do not exist for SOC calculations. Setting spin to 0."
            )
            spin = None

        d["window"] = kwargs.pop("window", 3)
        d["mark_fermi_level"] = kwargs.pop("mark_fermi_level", fermi_color)
        d["mark_gap"] = kwargs.pop("mark_gap", True)

        self.set_energy_reference(reference, self.soc)
        if bandpath != None:
            spectrum = self.get_mlk_spectrum(bandpath)
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

        m = {}
        m["spin"] = self.spin2index(spin)
        m["mode"] = kwargs.pop("mode", "lines")
        m["interpolate"] = kwargs.pop("interpolate", False)

        return axargs, kwargs, d, m

    def plot_contribution_of_one_atom(
        self, index, l="tot", color="crimson", axes=None, main=True, **kwargs
    ):
        axargs, kwargs, bsargs, mlkargs = self._process_kwargs(**kwargs)
        spectrum = bsargs["spectrum"]
        con = spectrum.get_atom_contribution(index, l)
        cmap = self.color_to_alpha_cmap(color)
        norm = Normalize(vmin=0.0, vmax=1.0)

        with AxesContext(ax=axes, main=main, **axargs) as axes:
            bs = BandStructurePlot(main=main, **bsargs)
            axes = bs.draw()
            x, y = bs.xy
            mlk = MullikenBandStructurePlot(
                x=x, y=y, con=con, cmap=cmap, norm=norm, **mlkargs
            )
            axes = mlk.draw()
        return axes

    def plot_one_species(
        self, symbol, l="tot", axes=None, color="crimson", main=True, **kwargs
    ):

        axargs, kwargs, bsargs, mlkargs = self._process_kwargs(**kwargs)
        spectrum = bsargs["spectrum"]
        cmap = self.color_to_alpha_cmap(color)
        norm = Normalize(vmin=0, vmax=1.0)
        con = spectrum.get_species_contribution(symbol, l=l)

        with AxesContext(ax=axes, main=main, **axargs) as axes:
            bs = BandStructurePlot(main=main, **bsargs)
            axes = bs.draw()
            x, y = bs.xy
            mlk = MullikenBandStructurePlot(
                x=x, y=y, con=con, cmap=cmap, norm=norm, **mlkargs
            )
            axes = mlk.draw()

        return axes

    def plot_all_species(
        self, symbols="all", l="tot", axes=None, colors=[], main=True, **kwargs
    ):
        axargs, kwargs, bsargs, mlkargs = self._process_kwargs(**kwargs)
        spectrum = bsargs["spectrum"]

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
            bs = BandStructurePlot(main=main, **bsargs)
            axes = bs.draw()
            x, y = bs.xy
            for i, (s, c, m) in enumerate(scm):
                cmap = self.color_to_alpha_cmap(c)
                con = spectrum.get_species_contribution(s, l=l)
                norm = Normalize(vmin=0, vmax=1.0)
                mlk = MullikenBandStructurePlot(
                    x=x, y=y, con=con, cmap=cmap, norm=norm, **mlkargs
                )
                axes = mlk.draw()
                handles.append(Line2D([0], [0], color=c, label=s, lw=1.5))
            axes.legend(
                handles=handles,
                frameon=True,
                fancybox=False,
                borderpad=0.4,
                loc="upper right",
            )

        return axes

    def plot_gradient_contributions(
        self, con1, con2, axes=None, colors=["blue", "red"], main=True, **kwargs
    ):
        axargs, kwargs, bsargs, mlkargs = self._process_kwargs(**kwargs)

        assert len(colors) == 2, "You can only specify 2 colors."

        con = con2 - con1
        cmap = LinearSegmentedColormap.from_list("", colors)
        my_cmap = cmap(np.arange(cmap.N))
        my_cmap = ListedColormap(my_cmap)
        norm = Normalize(vmin=-1.0, vmax=1.0)

        with AxesContext(ax=axes, main=main, **axargs) as axes:
            bs = BandStructurePlot(main=main, **bsargs)
            axes = bs.draw()
            x, y = bs.xy
            mlk = MullikenBandStructurePlot(
                x=x, y=y, con=con, cmap=cmap, norm=norm, **mlkargs
            )
            axes = mlk.draw()
            clb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes)
            clb.set_ticks([-1, 1])
            clb.set_alpha(1)  # alpha leads to weird stripes in color bars
            clb.set_ticklabels([con1.get_latex_symbol(), con2.get_latex_symbol()])

        return axes

    def plot_majority_contributions(
        self, list_of_contributions=[], axes=None, colors=[], main=True, **kwargs
    ):
        axargs, kwargs, bsargs, mlkargs = self._process_kwargs(**kwargs)
        scale_width = mlkargs.pop("scale_width", 2)

        if list_of_contributions == []:
            spectrum = bsargs["spectrum"]
            symbols = set(self.structure.symbols)
            list_of_contributions = [
                spectrum.get_species_contribution(c) for c in symbols
            ]

        assert (
            type(list_of_contributions) == list
        ), "You have to specify the contributions as a list."

        if colors == []:
            cmap = plt.cm.get_cmap("tab10")
            colors = [cmap(c) for c in np.linspace(0, 1, len(list_of_contributions))]
        assert len(colors) == len(
            list_of_contributions
        ), "Number of colors does not match number of contributions."

        con1 = list_of_contributions[0]
        con = np.zeros(con1.contribution.shape)
        for i, s, j in np.ndindex(con.shape):
            # at each kpoint i, each spin s, each state j, compare which value is largest
            l = [c.contribution[i, s, j] for c in list_of_contributions]
            k = l.index(max(l))
            # the index of the largest value is assigned to this point
            con[i, s, j] = k + 1
        con = MullikenContribution("Uh?", con, "eeeh")
        symbols = [c.get_latex_symbol() for c in list_of_contributions]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(
            [0.5 + j for j in range(len(colors))] + [len(colors) + 0.5], cmap.N
        )
        with AxesContext(ax=axes, main=main, **axargs) as axes:
            bs = BandStructurePlot(main=main, **bsargs)
            axes = bs.draw()
            x, y = bs.xy
            mlk = MullikenBandStructurePlot(
                x=x, y=y, con=con, cmap=cmap, norm=norm, scale_width=False, **mlkargs
            )
            axes = mlk.draw()
            clb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes)
            clb.set_ticks(range(1, len(colors) + 1))
            clb.set_ticklabels(symbols)

        return axes

    def plot_custom_contribution(
        self, contribution, axes=None, color="crimson", main=True, **kwargs
    ):
        axargs, kwargs, bsargs, mlkargs = self._process_kwargs(**kwargs)
        con = contribution
        cmap = self.color_to_alpha_cmap(color)
        norm = Normalize(vmin=0, vmax=1.0)

        with AxesContext(ax=axes, main=main, **axargs) as axes:
            bs = BandStructurePlot(main=main, **bsargs)
            axes = bs.draw()
            x, y = bs.xy
            mlk = MullikenBandStructurePlot(
                x=x, y=y, con=con, cmap=cmap, norm=norm, **mlkargs
            )
            mlk.draw()

        return axes

    def plot_all_angular_momenta(
        self, symbols="all", max_l="f", axes=None, colors=[], main=True, **kwargs
    ):
        axargs, kwargs, bsargs, mlkargs = self._process_kwargs(**kwargs)
        scale_width = mlkargs.pop("scale_width", 2)

        momenta = ("s", "p", "d", "f", "g", "h")
        momenta = dict(zip(momenta, range(len(momenta))))
        momenta = {k: v for k, v in momenta.items() if v <= momenta[max_l]}

        if symbols in ["all", "None", None, "All", []]:
            symbols = set([k.symbol for k in self.structure])
        else:
            symbols = set(symbols)

        if colors == []:
            cmap = plt.cm.get_cmap("tab10")
            colors = [cmap(c) for c in np.linspace(0, 1, 6)]
            colors = colors[: len(momenta)]

        lcons = [
            self.spectrum.get_group_contribution(symbols, l) for l in momenta.keys()
        ]
        con = np.zeros(lcons[0].contribution.shape)
        for i, s, j in np.ndindex(con.shape):
            # at each kpoint i, each spin s, each state j, compare which value is largest
            l = [c.contribution[i, s, j] for c in lcons]
            k = l.index(max(l))
            # the index of the largest value is assigned to this point
            con[i, s, j] = k + 1
        con = MullikenContribution("Uh?", con, "eeeh")

        cmap = ListedColormap(colors)
        norm = BoundaryNorm(
            [0.5 + j for j in range(len(colors))] + [len(colors) + 0.5], cmap.N
        )
        with AxesContext(ax=axes, main=main, **axargs) as axes:
            bs = BandStructurePlot(main=main, **bsargs)
            axes = bs.draw()
            x, y = bs.xy
            mlk = MullikenBandStructurePlot(
                x=x, y=y, con=con, cmap=cmap, norm=norm, scale_width=False, **mlkargs
            )
            axes = mlk.draw()
            clb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes)
            clb.set_ticks(range(1, len(colors) + 1))
            clb.set_ticklabels(list(momenta.keys()))

        return axes

    def plot(self, axes=None, color=mutedblack, main=True, **kwargs):
        axargs, kwargs, bsargs, _ = self._process_kwargs(**kwargs)
        with AxesContext(ax=axes, main=main, **axargs) as axes:
            bs = BandStructurePlot(main=main, **bsargs)
            axes = bs.draw()
            x, y = bs.xy
            axes.plot(x, y, color=color, **kwargs)
        return axes
