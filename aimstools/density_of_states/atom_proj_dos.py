from aimstools.misc import *
from aimstools.density_of_states.base import DOSBaseClass
from aimstools.density_of_states.species_proj_dos import SpeciesProjectedDOSMethods
from aimstools.density_of_states.utilities import (
    DOSSpectrum,
    Contribution,
    DOSPlot,
)

import numpy as np
import re
from ase.data.colors import jmol_colors
from ase.symbols import symbols2numbers


class AtomProjectedDOS(DOSBaseClass, SpeciesProjectedDOSMethods):
    def __init__(self, outputfile, soc=False) -> None:
        super().__init__(outputfile)
        assert any(
            x in ["atom-projected dos", "atom-projected dos tetrahedron"]
            for x in self.tasks
        ), "Atom-projected DOS was not specified as task in control.in ."

        self.soc = soc
        self.spin = "none" if self.control["spin"] != "collinear" else "collinear"
        if self.spin == "none":
            dosfiles = self.get_dos_files(soc=soc, spin="none").atom_proj_dos
            spectrum = self.read_dosfiles(list(zip(dosfiles, dosfiles)))
        if self.spin == "collinear":
            dosfiles_dn = self.get_dos_files(soc=soc, spin="dn").atom_proj_dos
            dosfiles_up = self.get_dos_files(soc=soc, spin="up").atom_proj_dos
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
        dos_per_atom = []
        energies = []
        nspins = 2 if self.spin == "collinear" else 1
        for i, atom in enumerate(self.structure):
            symbol = atom.symbol
            index = i + 1
            pattern = re.compile(r".*" + symbol + r"\d{0,3}" + str(index) + r".*")
            energies = []
            contributions = []
            for s in range(nspins):
                atom_file = [k[s] for k in dosfiles if re.match(pattern, str(k[s]))]
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
            dos_per_atom.append(con)
        spectrum = DOSSpectrum(energies, dos_per_atom, "atom")
        return spectrum

    def plot_one_atom(
        self, index, l="tot", axes=None, color=None, main=True, **kwargs,
    ):
        axargs, kwargs, dosargs = self._process_kwargs(**kwargs)
        assert index in range(
            len(self.structure)
        ), "The index {} was not found in the structure.".format(index)
        x = self.spectrum.energies
        con = self.spectrum.get_atom_contributions(index)
        symbol = con.symbol

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
