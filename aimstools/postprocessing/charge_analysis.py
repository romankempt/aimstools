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
            read = False
            for line in file.readlines():
                if "Performing Hirshfeld analysis" in line:
                    read = True
                    i = 0
                if read:
                    if "Atom" in line:
                        ats.append(
                            (int(line.split()[-2].strip(":")) - 1, line.split()[-1])
                        )
                    if "Hirshfeld charge        :" in line:
                        charges.append(float(line.split()[-1]))
                        i += 1
                        if i == self.structure.get_global_number_of_atoms():
                            read = False
        charges = dict(zip(ats, charges))
        return charges

    def sum_charges(self, fragment_indices=[]):
        """ Sums charges of given indices of atoms for the same species.
        
        Args:
            filter (list): List of atom indices. If None, all atoms of same species will be summed up.            
        """
        indices = {i: j for i, j in self.charges.keys()}

        fragment_filter = (
            [i[0] for i in self.charges.keys()]
            if fragment_indices == []
            else fragment_indices
        )

        sum_charges = {}
        for atom in fragment_filter:
            species = indices[atom]
            if species not in sum_charges.keys():
                sum_charges[species] = self.charges[(atom, species)]
            else:
                sum_charges[species] += self.charges[(atom, species)]
        return sum_charges
