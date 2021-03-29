from aimstools.density_of_states.total_dos import TotalDOS
from aimstools.misc import *
from aimstools.density_of_states.base import DOSBaseClass
from aimstools.density_of_states.atom_proj_dos import AtomProjectedDOS
from aimstools.density_of_states.utilities import DOSPlot, DOSContribution, DOSSpectrum

import numpy as np

import re


class SpeciesProjectedDOS(AtomProjectedDOS, TotalDOS, DOSBaseClass):
    def __init__(self, outputfile, soc=False) -> None:
        DOSBaseClass.__init__(self, outputfile)
        assert any(
            x in ["species-projected dos", "species-projected dos tetrahedron"]
            for x in self.tasks
        ), "Species-projected DOS was not specified as task in control.in ."
        self.soc = soc
        self.spin = "none" if self.control["spin"] != "collinear" else "collinear"

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
        atoms = self.structure.copy()
        self._spectrum = DOSSpectrum(
            atoms=atoms,
            energies=energies,
            contributions=dos_per_species,
            type="species",
            fermi_level=fermi_level,
            reference=reference,
            shift=shift,
        )
