from aimstools.misc import *
from aimstools.postprocessing import FHIAimsOutputReader
from aimstools.structuretools import Structure

import os

logger = logging.getLogger("root")


class HirshfeldReader(FHIAimsOutputReader):
    """ A simple class to evaluate Hirshfeld charge analysis from AIMS.
    Args:
        output (pathlib object): Directory of outputfile or outputfile.
    Attributes:
        charges (dict): Dictionary of (index, species) tuples and hirshfeld charges. 
        tot_charges (dict): Dictionary of species and summed hirshfeld charges.
    
    """

    def __init__(self, output):
        super().__init__(output)
        self.charges = self.read_charges()
        self.total_charges = self.sum_charges()

    def read_charges(self):
        with open(self.outputfile, "r") as file:
            ats = []
            charges = []
            spins = []
            read = False
            for line in file.readlines():
                if (
                    "Performing Hirshfeld analysis of fragment charges and moments."
                    in line
                ):
                    read = True
                    i = 0
                if read:
                    if "Atom" in line:
                        index, symbol = (
                            int(line.split()[-2].strip(":")) - 1,
                            line.split()[-1],
                        )
                        ats.append(index)
                    if "Hirshfeld charge        :" in line:
                        charges.append(float(line.split()[-1]))
                    if "Hirshfeld second moments:" in line:
                        i += 1
                        if i == self.structure.get_global_number_of_atoms():
                            read = False
        charges = dict(zip(ats, charges))
        return charges

    def read_spins(self):
        with open(self.outputfile, "r") as file:
            ats = []
            spins = []
            read = False
            for line in file.readlines():
                if (
                    "Performing Hirshfeld analysis of fragment charges and moments."
                    in line
                ):
                    read = True
                    i = 0
                if read:
                    if "Atom" in line:
                        index, symbol = (
                            int(line.split()[-2].strip(":")) - 1,
                            line.split()[-1],
                        )
                        ats.append(index)
                    if "Hirshfeld spin moment   :" in line:
                        spins.append(float(line.split()[-1]))
                    if "Hirshfeld second moments:" in line:
                        i += 1
                        if i == self.structure.get_global_number_of_atoms():
                            read = False
        spins = dict(zip(ats, spins))
        return spins

    def sum_charges(self, fragment_indices=[]):
        """ Sums charges of given indices of atoms for the same species.
        
        Args:
            filter (list): List of atom indices. If None, all atoms of same species will be summed up.            
        """

        assert all(
            [k >= 0 for k in fragment_indices]
        ), "Indices of fragments should be non-negative."

        atom_indices = [
            j if j >= 0 else len(self.charges.keys()) + j for j in self.charges.keys()
        ]
        if fragment_indices != []:
            atom_indices = [j for j in atom_indices if j in fragment_indices]
        atoms = self.structure[atom_indices]
        species = set(atoms.get_chemical_symbols())

        sum_charges = dict(zip(species, [0] * len(species)))
        for s in species:
            for atom in atoms:
                if atom.symbol == s:
                    sum_charges[s] += self.charges[atom.index]
        return sum_charges
