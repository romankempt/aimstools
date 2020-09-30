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
    """ Extends the ase.atoms.Atoms class with some specific functions.
    
    Args:
        geometry (str): Path to structure file (.cif, .xyz ..) or atoms object.

    Attributes:
        atoms (Atoms): ASE atoms object.
        atom_indices (dict): Dictionary of atom index and label.
        species (dict): Dictionary of atom labels and counts.
        sg (spacegroup): Spglib spacegroup object.
        lattice (str): Description of Bravais lattice.
    """

    def __init__(self, geometry):
        if type(geometry) == ase.atoms.Atoms:
            self.atoms = geometry
        elif Path(geometry).is_file():
            try:
                self.atoms = ase.io.read(geometry)
            except:
                logging.error("Input structure not recognised.")
        assert type(self.atoms) == ase.atoms.Atoms, "Atoms not read correctly."
        self.atom_indices = dict(
            zip(range(1, len(self.atoms) + 1), list(self.atoms.symbols))
        )
        keys = list(set(self.atom_indices.values()))
        numbers = []
        for key in keys:
            numbers.append(list(self.atom_indices.values()).count(key))
        try:
            self.sg = ase.spacegroup.get_spacegroup(self.atoms, symprec=1e-2)
        except:
            self.sg = ase.spacegroup.Spacegroup(1)
        self.lattice = self.atoms.cell.get_bravais_lattice().crystal_family
        self.species = dict(zip(keys, numbers))

    def __repr__(self):
        return self.atoms.get_chemical_formula()

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

    def standardize(self, atoms=None, to_primitive=True, symprec=1e-4):
        """ Wrapper of the spglib standardize() function with extra features.

        For 2D systems, the non-periodic axis is enforced as the z-axis.
        
        Args:
            to_primitive (bool): Reduces to primitive cell or not.
            symprec (float): Precision to determine new cell.

        Note:
            The combination of to_primitive=True and a larger value of symprec (1e-3) can be used to symmetrize a structure.
        """
        if atoms != None:
            atoms = atoms.copy()
        else:
            atoms = self.atoms.copy()
        pbc1 = self.find_nonperiodic_axes(atoms)
        lattice, positions, numbers = (
            atoms.get_cell(),
            atoms.get_scaled_positions(),
            atoms.numbers,
        )
        cell = (lattice, positions, numbers)
        newcell = spglib.standardize_cell(
            cell, to_primitive=to_primitive, no_idealize=False, symprec=symprec
        )
        if newcell == None:
            logging.error("Cell could not be standardized.")
            return None
        else:
            atoms = ase.Atoms(
                scaled_positions=newcell[1],
                numbers=newcell[2],
                cell=newcell[0],
                pbc=self.atoms.pbc,
            )
            pbc2 = self.find_nonperiodic_axes(atoms)
            if pbc1 != pbc2:
                old = [k for k, v in pbc1.items() if v]
                new = [k for k, v in pbc2.items() if v]
                assert len(old) == len(
                    new
                ), "Periodicity changed due to standardization."
                if len(new) == 2:
                    npbcax = list(set([0, 1, 2]) - set(new))[0]
                    atoms = ase.geometry.permute_axes(atoms, new + [npbcax])
                    assert self.is_2d(atoms), "Permutation to 2D not working."
            self.atoms = atoms
            self.lattice = atoms.cell.get_bravais_lattice().crystal_family

    def enforce_2d(self, atoms=None):
        """ Enforces a special representation of a two-dimensional system.
        
        Sets all z-components of the lattice basis to zero and adds vacuum space such that the layers are centered in the unit cell.

        Returns:
            atoms: Modified atoms object.
        """
        if atoms != None:
            atoms = atoms.copy()
        else:
            atoms = self.atoms.copy()
        logging.info("Enforcing standardized 2D representation ...")
        try:
            self.standardize(atoms)
            assert len(atoms) == len(
                self.atoms
            ), "Number of atoms changed due to standardization. Reverting."
        except:
            self.atoms = atoms.copy()
            logging.warning("Number of atoms changed due to standardization.")
            logging.warning("Reverting spglib standardization.")
        finally:
            logging.warning(
                "Using ASE standardization, check if cell is set correctly."
            )
            rcell, Q = atoms.cell.standard_form()
            atoms.set_cell(rcell)
            atoms.set_positions((Q @ atoms.positions.T).T)
        atoms = self.recenter(atoms)
        if self.is_2d(atoms) != True:
            logging.warning("Enforcing 2D system failed.")
            logging.warning(
                "This is typically the case, if the cell orientation changes because of stupid crystallographic conventions."
            )
            return self.atoms
        else:
            return atoms

    def recenter(self, atoms=None):
        """ Recenters atoms to be in the unit cell, with vacuum on both sides.

        The unit cell length c is always chosen such that it is larger than a and b.
        
        Returns:
            atoms : modified atoms object.

        Note:
            The ase.atoms.center() method is supposed to do that, but sometimes separates the layers. I didn't find a good way to circumvene that.

         """
        if atoms != None:
            atoms = atoms.copy()
        else:
            atoms = self.atoms.copy()
        atoms.wrap(pretty_translation=True)
        atoms.center(axis=(2))
        mp = atoms.get_center_of_mass(scaled=False)
        cp = (atoms.cell[0] + atoms.cell[1] + atoms.cell[2]) / 2
        pos = atoms.get_positions(wrap=False)
        pos[:, 2] += np.abs((mp - cp))[2]
        for z in range(pos.shape[0]):
            lz = atoms.cell.lengths()[2]
            if pos[z, 2] >= lz:
                pos[z, 2] -= lz
            if pos[z, 2] < 0:
                pos[z, 2] += lz
        atoms.set_positions(pos)
        newcell, newpos, newscal, numbers = (
            atoms.get_cell(),
            atoms.get_positions(wrap=False),
            atoms.get_scaled_positions(wrap=False),
            atoms.numbers,
        )
        z_pos = newpos[:, 2]
        span = np.max(z_pos) - np.min(z_pos)
        newcell[0, 2] = newcell[1, 2] = newcell[2, 0] = newcell[2, 1] = 0.0
        newcell[2, 2] = span + 100.0
        axes = [0, 1, 2]
        lengths = np.linalg.norm(newcell, axis=1)
        order = [x for x, y in sorted(zip(axes, lengths), key=lambda pair: pair[1])]
        while True:
            if (order == [0, 1, 2]) or (order == [1, 0, 2]):
                break
            newcell[2, 2] += 10.0
            lengths = np.linalg.norm(newcell, axis=1)
            order = [x for x, y in sorted(zip(axes, lengths), key=lambda pair: pair[1])]
        newpos = newscal @ newcell
        newpos[:, 2] = z_pos
        atoms = ase.Atoms(
            positions=newpos, numbers=numbers, cell=newcell, pbc=atoms.pbc
        )
        return atoms

    def is_2d(self, atoms=None):
        """ Evaluates if given structure is qualitatively two-dimensional.

        Note:
            A 2D structure is considered 2D if only the z-axis is non-periodic.
        
        Returns:
            bool: 2D or not to 2D, that is the question.
        """
        if atoms != None:
            atoms = atoms.copy()
        else:
            atoms = self.atoms.copy()
        pbcax = self.find_nonperiodic_axes(atoms)
        if list(pbcax.values()) == [True, True, False]:
            return True
        else:
            return False

    def find_nonperiodic_axes(self, atoms=None):
        """ Evaluates if given structure is qualitatively periodic along certain lattice directions.

        Note:
            An axis is considered non-periodic if:\n
            - the structure consists of more than one fragment (non-continuos bonding)\n
            - two closest fragments are sepearted by more than 30.0 Anström\n

        Returns:
            dict: Axis : Bool pairs.
        """

        if atoms != None:
            atoms = atoms.copy()
        else:
            atoms = self.atoms.copy()

        sc = ase.build.make_supercell(atoms, 2 * np.identity(3), wrap=True)
        fragments = self.find_fragments(sc)
        crit1 = True if len(fragments) > 1 else False
        pbc = dict(zip([0, 1, 2], [True, True, True]))
        if crit1:
            for axes in (0, 1, 2):
                spans = []
                for index, tup in fragments.items():
                    start = np.min(tup[1].get_positions()[:, axes])
                    cm = tup[1].get_center_of_mass()[axes]
                    spans.append([start, cm])
                spans = sorted(spans, key=lambda x: x[0])
                for j in range(len(spans) - 1):
                    nd = spans[j + 1][1] - spans[j][1]
                    if nd >= 30.0:
                        pbc[axes] = False
                        break
        return pbc

    def hexagonal_to_rectangular(self, atoms=None):
        """ Changes hexagonal / trigonal unit cell to equivalent rectangular representation.
        
        Returns:
            atoms: ase.atoms.Atoms object
        """

        if atoms == None:
            atoms = self.atoms.copy()

        self.standardize(atoms)
        atoms = self.atoms.copy()
        old = atoms.cell.copy()
        a = old[0, :]
        b = old[1, :]
        c = old[2, :]
        newb = 2 * b + a
        newcell = np.array([a, newb, c])
        new = ase.build.make_supercell(atoms, [[3, 0, 0], [0, 3, 0], [0, 0, 1]])
        new.set_cell(newcell, scale_atoms=False)
        spos = new.get_scaled_positions(wrap=False)
        inside = np.where(
            (spos[:, 0] >= 0)
            & (spos[:, 0] < 0.9999)
            & (spos[:, 1] >= 0)
            & (spos[:, 1] < 0.9999)
        )[0]
        new = new[inside]
        return new

