import numpy as np
from pathlib import Path as Path
import re
import ase.io
import os, sys
from AIMS_tools.postprocessing import postprocess
from AIMS_tools.structuretools import structure


class hirshfeld(postprocess):
    """ A simple class to evaluate Hirshfeld charge analysis from AIMS.

    Args:
        outputfile (str): Path to outputfile.

    Attributes:
        charges (dict): Dictionary of (index, species) tuples and hirshfeld charges. 
        tot_charges (dict): Dictionary of species and summed hirshfeld charges.
    
    """

    def __init__(self, outputfile, get_SOC=True, spin=None):
        super().__init__(outputfile, get_SOC=get_SOC, spin=spin)
        self.charges = self.read_charges()
        self.tot_charges = self.sum_charges()

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
                        if i == self.structure.atoms.get_global_number_of_atoms():
                            read = False
        charges = dict(zip(ats, charges))
        return charges

    def sum_charges(self, fragment_filter=[]):
        """ Sums charges of given indices of atoms for the same species.
        
        Args:
            filter (list): List of atom indices. If None, all atoms of same species will be summed up.
        
        Example:
            >>> from AIMS_tools.structuretools import structure
            >>> from AIMS_tools import hirshfeld
            >>> hs = hirshfeld.hirshfeld("outputfile")
            >>> frag1 = hs.structure.fragments[0][0]
            >>> hs.sum_charges(frag1)
            
        """
        indices = {i: j for i, j in self.charges.keys()}

        fragment_filter = (
            [i[0] for i in self.charges.keys()]
            if fragment_filter == []
            else fragment_filter
        )

        sum_charges = {}
        for atom in fragment_filter:
            species = indices[atom]
            if species not in sum_charges.keys():
                sum_charges[species] = self.charges[(atom, species)]
            else:
                sum_charges[species] += self.charges[(atom, species)]
        return sum_charges

