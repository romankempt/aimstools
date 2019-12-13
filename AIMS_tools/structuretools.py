from ase import neighborlist, build
import ase.io
import sys, os, math
from pathlib import Path as Path
import numpy as np
import networkx as nx
import spglib

Angstroem_to_bohr = 1.889725989


class structure:
    """ A base class for structure analysis and manipulation relying on the ASE.
    
    Args:
        geometryfile (str): Path to structure file (.cif, .xyz ..).

    Attributes:
        atoms (Atoms): ASE atoms object.
        atom_indices (dict): Dictionary of atom index and label.
        species (dict): Dictionary of atom labels and counts.
        fragments (dict): Dictionary of (index: (original_index, atoms)) pairs.
        rec_cell (array): Reciprocal lattice vectors in 2 pi/bohr.
        rec_cell_lengths (numpy array): Reciprocal lattice vector lengths in 2 pi/bohr. 
        attributes (dict): Collection of attributes.
    """

    def __init__(self, geometryfile):
        self.atoms = ase.io.read(geometryfile)
        self.attributes = {}
        self.find_fragments()
        self.rec_cell = (
            self.atoms.get_reciprocal_cell() * 2 * math.pi / Angstroem_to_bohr
        )
        self.rec_cell_lengths = ase.cell.Cell.new(
            self.rec_cell
        ).lengths()  # converting to atomic units 2 pi/bohr
        self.atom_indices = {}
        i = 1  # index to run over atoms
        with open(geometryfile, "r") as file:
            for line in file.readlines():
                if "atom" in line:
                    self.atom_indices[i] = line.split()[-1]
                    i += 1
        keys = list(set(self.atom_indices.values()))
        numbers = []
        for key in keys:
            numbers.append(list(self.atom_indices.values()).count(key))
        self.species = dict(zip(keys, numbers))

    def find_fragments(self):
        """ Finds unconnected structural fragments by constructing
        the first-neighbor topology matrix and the resulting graph
        of connected vortices. 
        
        Note:
            Requires networkx library.

        """

        atoms = self.atoms
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
        con_tuples = list(nx.connected_components(graph))  # beauty of graph theory

        fragments = {}
        i = 0
        for tup in con_tuples:
            fragments[i] = ase.Atoms()
            indices = []
            for entry in tup:
                fragments[i].append(atoms[entry])
                indices.append(entry)
            fragments[i].cell = atoms.cell
            fragments[i].pbc = atoms.pbc
            fragments[i] = (indices, fragments[i])
            i += 1
        self.attributes["number_of_fragments"] = len(fragments)
        self.fragments = fragments

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

        """
        av_c_1 = np.average(fragment1.get_positions()[:, 2])
        av_c_2 = np.average(fragment2.get_positions()[:, 2])
        av_c = np.abs(av_c_2 - av_c_1)
        self.attributes["average_layer_distance"] = av_c

        if av_c_2 > av_c_1:
            a = np.min(fragment2.get_positions()[:, 2])
            b = np.max(fragment1.get_positions()[:, 2])
            print(a, b)
            int_d = a - b
            self.attributes["interstitial_distance"] = int_d
        elif av_c_1 > av_c_2:
            a = np.min(fragment1.get_positions()[:, 2])
            b = np.max(fragment2.get_positions()[:, 2])
            print(a, b)
            int_d = a - b
            self.attributes["interstitial_distance"] = int_d

    def standardize(self):
        """ Standardizes to the conventional unit cell employing the space group library. """
        lattice, positions, numbers = (
            self.atoms.get_cell(),
            self.atoms.get_scaled_positions(),
            self.atoms.numbers,
        )
        cell = (lattice, positions, numbers)
        newcell = spglib.standardize_cell(cell, to_primitive=False, no_idealize=False)
        if newcell == None:
            sys.exit("Cell could not be standardized.")
        else:
            self.atoms = ase.Atoms(
                scaled_positions=newcell[1],
                numbers=newcell[2],
                cell=newcell[0],
                pbc=self.atoms.pbc,
            )

    def enforce_2d(self):
        """ Enforces a 2D system by setting all z-components of the lattice basis to zero
        and puts an orthogonal c-axis of 100 Angström."""
        newcell = self.atoms.cell
        scaled_positions = self.atoms.get_scaled_positions()
        z_positions = self.atoms.get_positions()[:, 2]
        newcell[0, 2] = newcell[1, 2] = newcell[2, 0] = newcell[2, 1] = 0
        newcell[2, 2] = 100.0
        self.atoms.cell = newcell
        self.atoms.pbc = [True, True, False]
        self.atoms.positions = scaled_positions * self.atoms.cell.lengths()
        self.atoms.positions[:, 2] = z_positions


