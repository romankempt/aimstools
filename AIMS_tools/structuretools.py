import sys, os, math
from pathlib import Path as Path
import numpy as np
import networkx as nx
import spglib

import ase.io
import ase.spacegroup
from ase import neighborlist, build

from AIMS_tools.misc import *

from collections import namedtuple


class structure:
    """ A base class for structure analysis and manipulation relying on the ASE.
    
    Args:
        geometryfile (str): Path to structure file (.cif, .xyz ..).

    Attributes:
        atoms (Atoms): ASE atoms object.
        atom_indices (dict): Dictionary of atom index and label.
        species (dict): Dictionary of atom labels and counts.
        sg (spacegroup): Spglib spacegroup object.
        lattice (str): Description of Bravais lattice.
    """

    def __init__(self, geometryfile):
        self.atoms = ase.io.read(geometryfile)
        self.atom_indices = {}
        i = 1  # index to run over atoms
        with open(geometryfile, "r") as file:
            for line in file.readlines():
                if ("atom" in line) and ("hessian" not in line):
                    self.atom_indices[i] = line.split()[-1]
                    i += 1
        keys = list(set(self.atom_indices.values()))
        numbers = []
        for key in keys:
            numbers.append(list(self.atom_indices.values()).count(key))
        self.sg = ase.spacegroup.get_spacegroup(self.atoms, symprec=1e-4)
        self.lattice = self.__get_lattice()
        self.species = dict(zip(keys, numbers))
        self.fragments = self.find_fragments(self.atoms)

    def __repr__(self):
        return self.atoms.get_chemical_formula()

    def __get_lattice(self):
        nr = self.sg.no
        if nr <= 2:
            return "triclinic"
        elif nr <= 15:
            return "monoclinic"
        elif nr <= 74:
            return "orhothombic"
        elif nr <= 142:
            return "tetragonal"
        elif nr <= 167:
            return "trigonal"
        elif nr <= 194:
            return "hexagonal"
        else:
            return "cubic"

    def find_fragments(self, atoms):
        """ Finds unconnected structural fragments by constructing
        the first-neighbor topology matrix and the resulting graph
        of connected vortices. 

        Args:
            atoms (atoms): ASE atoms object.
        
        Note:
            Requires networkx library.
        
        Returns:
            dict: Dictionary with named tuples of indices and atoms, sorted by average z-value.

        """

        nl = neighborlist.NeighborList(
            ase.neighborlist.natural_cutoffs(atoms),
            self_interaction=False,
            bothways=True,
        )
        nl.update(atoms)
        connectivity_matrix = nl.get_connectivity_matrix(sparse=False)

        con_tuples = {}  # connected first neighbors
        for row in range(connectivity_matrix.shape[0]):
            con_tuples[row] = []
            for col in range(connectivity_matrix.shape[1]):
                if connectivity_matrix[row, col] == 1:
                    con_tuples[row].append(col)

        pairs = []  # cleaning up the first neighbors
        for index in con_tuples.keys():
            for value in con_tuples[index]:
                if index > value:
                    pairs.append((index, value))
                elif index <= value:
                    pairs.append((value, index))
        pairs = set(pairs)

        graph = nx.from_edgelist(pairs)  # converting to a graph
        con_tuples = list(
            nx.connected_components(graph)
        )  # graph theory can be pretty handy

        fragments = {}
        i = 0
        for tup in con_tuples:
            fragment = namedtuple("fragment", ["indices", "atoms"])
            ats = ase.Atoms()
            indices = []
            for entry in tup:
                ats.append(atoms[entry])
                indices.append(entry)
            ats.cell = atoms.cell
            ats.pbc = atoms.pbc
            fragments[i] = fragment(indices, ats)
            i += 1
        fragments = {
            k: v
            for k, v in sorted(
                fragments.items(),
                key=lambda item: np.average(item[1][1].get_positions()[:, 2]),
            )
        }
        return fragments

    def calculate_interlayer_distances(self, fragment1, fragment2):
        """ A specialised method for 2D layered systems to determine the interlayer distance and interstitial distance.

        The average layer distance is defined as the difference between the average z-components
        of two fragments:

        .. math:: d_{int} = \sum_i^{N_1} z_i /{N_1} - \sum_j^{N_2} z_j / {N_2}

        The interstitial distance is defined as the difference between the z-components of the
        two closest atoms of two fragments:

        .. math:: d_{isd} = min( z_1, z_2, ..., z_{N_1}) - max( z_1, z_2, ..., z_{N_2})

        Note:
            This method also works for non-planar (curved) systems or interdented systems
            by calculating average distances of bonded fragments. The results do not have
            to make sense though.
        
        Args:
            fragment1, fragment2 (Atoms): ASE atoms objects.

        """
        av_c_1 = np.average(fragment1.get_positions()[:, 2])
        av_c_2 = np.average(fragment2.get_positions()[:, 2])
        av_c = np.abs(av_c_2 - av_c_1)
        logging.info("Average layer distance: \t {: 10.3f} Angström".format(av_c))

        if av_c_2 > av_c_1:
            a = np.min(fragment2.get_positions()[:, 2])
            b = np.max(fragment1.get_positions()[:, 2])
            int_d = a - b
            logging.info("Interstitial distance: \t {: 10.3f} Angström".format(int_d))
        elif av_c_1 > av_c_2:
            a = np.min(fragment1.get_positions()[:, 2])
            b = np.max(fragment2.get_positions()[:, 2])
            int_d = a - b
            logging.info("Interstitial distance: \t {: 10.3f} Angström".format(int_d))

    def standardize(self, to_primitive=False, symprec=1e-4):
        """ Wrapper of the spglib standardize() function.
        
        Args:
            to_primitive (bool): Reduces to primitive cell or not.
            symprec (float): Precision to determine new cell.

        Note:
            The combination of to_primitive=True and a larger value of symprec (1e-3) can be used to symmetrize a structure.
        """
        lattice, positions, numbers = (
            self.atoms.get_cell(),
            self.atoms.get_scaled_positions(),
            self.atoms.numbers,
        )
        cell = (lattice, positions, numbers)
        newcell = spglib.standardize_cell(
            cell, to_primitive=to_primitive, no_idealize=False, symprec=symprec
        )
        if newcell == None:
            logging.error("Cell could not be standardized.")
            return None
        else:
            self.atoms = ase.Atoms(
                scaled_positions=newcell[1],
                numbers=newcell[2],
                cell=newcell[0],
                pbc=self.atoms.pbc,
            )

    def enforce_2d(self):
        """ Enforces a 2D system.
        
        Sets all z-components of the lattice basis to zero and adds vacuum space.

        Args:
            atoms (atoms): ASE atoms object.

        Returns:
            atoms: Modified atoms object.
        """
        self.standardize()
        atoms = self.atoms
        atoms.center(axis=(2))
        mp = atoms.get_center_of_mass()
        cp = (atoms.cell[0] + atoms.cell[0] + atoms.cell[0]) / 2
        print("mp: ", mp)
        print("cp: ", cp)
        atoms.wrap(pretty_translation=True)
        pos = atoms.get_positions()
        pos[:, 2] = (mp - cp)[2]
        print("pos", pos)
        atoms.set_positions(pos)
        newcell, positions, numbers = (
            self.atoms.get_cell(),
            self.atoms.get_positions(),
            self.atoms.numbers,
        )
        scaled_positions = atoms.get_scaled_positions()
        z_positions = positions[:, 2]
        span = np.max(z_positions) - np.min(z_positions)
        newcell[0, 2] = newcell[1, 2] = newcell[2, 0] = newcell[2, 1] = 0.0
        newcell[2, 2] = span + 100.0
        newlengths = ase.cell.Cell.ascell(newcell).lengths()
        newpos = scaled_positions * newlengths
        newpos[:, 2] = z_positions
        print("pos", newpos)
        atoms = ase.Atoms(
            positions=newpos, numbers=numbers, cell=newcell, pbc=atoms.pbc,
        )
        assert self.is_2d(atoms) == True, "Enforcing 2D system failed."
        return atoms

    def is_2d(self, atoms):
        """ Evaluates if given structure is qualitatively two-dimensional.

        Note:
            A 2D structure has to fulfill three criterions:\n
            - more than one distinct unbonded fragments\n
            - a vacuum gap between at least one pair of closest fragments of at least 30 Angström\n
            - continouos in-plane connectivity within 30 Angström and periodicity\n
        
            The current code might fail for large structures with a small vacuum gap. Please report any
            cases where the result is wrong.

        Returns:
            bool: 2D or not to 2D, that is the question.
        """

        crit_1 = False  # criterion of distinct fragments
        crit_2 = False  # criterion of separated fragments
        crit_3 = False  # criterion of 2D periodicity and connectivity
        sc = ase.build.make_supercell(atoms, 2 * np.identity(3), wrap=True)
        fragments = self.find_fragments(sc)
        if len(fragments) > 1:
            crit_1 = True
        av_z = []
        for index, tup in fragments.items():
            zval = tup[1].get_positions()[:, 2]
            zval = np.average(zval)
            av_z.append(zval)
        av_z = sorted(av_z)
        for j in range(len(av_z) - 1):
            nearest_d = av_z[j + 1] - av_z[j]
            if nearest_d >= 30.0:
                crit_2 = True
                break

        if len(fragments) > 1:
            av_xy = []
            for index, tup in fragments.items():
                xy = tup[1].get_positions()[:, [0, 1]]
                xy = np.average(np.linalg.norm(xy, axis=1))
                av_xy.append(xy)
            av_xy = sorted(av_xy)
            for j in range(len(av_xy) - 1):
                nearest_d = av_xy[j + 1] - av_xy[j]
                if nearest_d >= 30.0:
                    break
            else:
                crit_3 = True

        if crit_1 * crit_2 * crit_3 == True:
            return True
        else:
            return False

