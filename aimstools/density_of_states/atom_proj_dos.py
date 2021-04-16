from aimstools.misc import *
from aimstools.density_of_states.base import DOSBaseClass
from aimstools.density_of_states.total_dos import TotalDOS
from aimstools.density_of_states.species_proj_dos import SpeciesProjectedDOS
from aimstools.density_of_states.utilities import DOSSpectrum

import numpy as np
import re


class AtomProjectedDOS(SpeciesProjectedDOS, TotalDOS, DOSBaseClass):
    def __init__(self, outputfile, soc=False) -> None:
        DOSBaseClass.__init__(self, outputfile)
        assert any(
            x
            in [
                "atom-projected dos",
                "atom-projected dos tetrahedron",
                "species-projected dos",
                "species-projected dos tetrahedron",
            ]
            for x in self.tasks
        ), "Atom-projected DOS was not specified as task in control.in ."
        self.soc = soc
        self.spin = "none" if self.control["spin"] != "collinear" else "collinear"
        self._spectrum = self.set_spectrum(None)

    def __repr__(self):
        return "{}(outputfile={}, spin_orbit_coupling={})".format(
            self.__class__.__name__, repr(self.outputfile), self.soc
        )

    def _read_dosfiles(self):
        if self.spin == "none":
            dosfiles = self.get_dos_files(soc=self.soc, spin="none").atom_proj_dos
            dosfiles = list(zip(dosfiles, dosfiles))
        if self.spin == "collinear":
            dosfiles_dn = self.get_dos_files(soc=self.soc, spin="dn").atom_proj_dos
            dosfiles_up = self.get_dos_files(soc=self.soc, spin="up").atom_proj_dos
            dosfiles = list(zip(dosfiles_dn, dosfiles_up))
        dos_per_atom = []
        energies = []
        nspins = 2 if self.spin == "collinear" else 1
        for i, atom in enumerate(self.structure):
            symbol = atom.symbol
            index = i + 1
            pattern = re.compile(r".*" + symbol + "{0:04d}".format(index) + r".*")
            energies = []
            contributions = []
            for s in range(nspins):
                atom_file = [k[s] for k in dosfiles if re.search(pattern, str(k[s]))]
                assert (
                    len(atom_file) == 1
                ), "Multiple atom-projected dos files found for same atom. Something must have gone wrong. Found: {}".format(
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
            dos_per_atom.append((i, contributions))
        self._dos = (energies, dos_per_atom)

    def set_spectrum(self, reference=None):
        if self.dos == None:
            self._read_dosfiles()
        energies, dos_per_atom = self.dos
        self.set_energy_reference(reference, self.soc)
        fermi_level = self.fermi_level.soc if self.soc else self.fermi_level.scalar
        reference, shift = self.energy_reference
        atoms = self.structure.atoms
        self._spectrum = DOSSpectrum(
            atoms=atoms,
            energies=energies,
            contributions=dos_per_atom,
            type="atom",
            fermi_level=fermi_level,
            reference=reference,
            shift=shift,
        )

    @property
    def spectrum(self):
        ":class:`aimstools.density_of_states.utilities.DOSSpectrum`."
        if self._spectrum == None:
            self.set_spectrum(None)
        return self._spectrum
