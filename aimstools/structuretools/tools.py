from aimstools.misc import *
from pathlib import Path as Path
import numpy as np
import networkx as nx

import ase.io
from ase import neighborlist, build
from ase.data import covalent_radii

from aimstools.misc import *

from collections import namedtuple


def find_fragments(atoms, scale=1.0) -> list:
    """Finds unconnected structural fragments by constructing
    the first-neighbor topology matrix and the resulting graph
    of connected vertices.

    Args:
        atoms: :class:`~ase.atoms.Atoms` or :class:`~aimstools.structuretools.structure.Structure`.
        scale: Scaling factor for covalent radii.

    Note:
        Requires networkx library.

    Returns:
        list: NamedTuple with indices and atoms object.

    """

    atoms = atoms.copy()
    radii = scale * covalent_radii[atoms.get_atomic_numbers()]
    nl = neighborlist.NeighborList(radii, skin=0, self_interaction=False, bothways=True)
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
    fragments = [
        v
        for k, v in sorted(
            fragments.items(),
            key=lambda item: np.average(item[1][1].get_positions()[:, 2]),
        )
    ]
    return fragments


def find_periodic_axes(atoms, vacuum_space=25, factor=1.1) -> dict:
    """Evaluates if given structure is qualitatively periodic along certain lattice directions.

    Args:
        atoms (ase.atoms.Atoms): ase.atoms.Atoms object.
        vacuum_space (float): Gaps larger than this are considered vacuum gaps.
        factor (float): Skin factor for neighborlist.

    Returns:
        dict: Axis : Bool pairs.
    """
    atoms = atoms.copy()
    pbc = dict(zip([0, 1, 2], atoms.pbc))
    is_layered, sublayers, indices = find_layers(atoms, factor=factor)
    if sublayers == None:
        return pbc

    fragment = namedtuple("fragment", ["indices", "atoms"])
    fragments = [fragment(i, a) for i, a in zip(indices, sublayers)]
    if is_layered:
        if len(fragments) == 1:
            a = fragments[0].atoms
            span = np.max(a.positions[:, 2]) - np.min(a.positions[:, 2])
            gap = a.cell.lengths()[2]
            pbc[2] = False if abs(gap - span) > vacuum_space else True
        else:
            for f1, f2 in zip(fragments[:-1], fragments[1:]):
                low1 = np.min(f1.atoms.positions[:, 2])
                low2 = np.min(f2.atoms.positions[:, 2])
                up1 = np.max(f1.atoms.positions[:, 2])
                up2 = np.max(f2.atoms.positions[:, 2])
                d = np.min(np.abs([up2 - low1, up1 - low2]))
                if d > vacuum_space:
                    pbc[2] = False
    return pbc


def hexagonal_to_rectangular(atoms) -> ase.atoms.Atoms:
    """Changes hexagonal / trigonal unit cell to equivalent rectangular representation.

    Cell must be in standard form.

    Args:
        atoms: ase.atoms.Atoms object

    Returns:
        atoms: ase.atoms.Atoms object
    """
    atoms = atoms.copy()
    assert atoms.cell.get_bravais_lattice().crystal_family in [
        "hexagonal",
        "trigonal",
    ], "This method only makes sense for trigonal or hexagonal systems."
    old = atoms.cell.copy()
    a = old[0, :]
    b = old[1, :]
    c = old[2, :]
    newb = 2 * b + a
    newcell = np.array([a, newb, c])
    new = ase.build.make_supercell(atoms, [[3, 0, 0], [0, 3, 0], [0, 0, 1]])
    new.set_cell(newcell, scale_atoms=False)
    spos = new.get_scaled_positions(wrap=False)
    spos = clean_matrix(spos)
    inside = np.where(
        (spos[:, 0] >= 0.0)
        & (spos[:, 0] < 0.9999999)
        & (spos[:, 1] >= 0.0)
        & (spos[:, 1] < 0.9999999)
    )[0]
    new = new[inside]
    return new


def clean_matrix(matrix, eps=1e-9):
    """ clean from small values"""
    matrix = np.array(matrix)
    for ij in np.ndindex(matrix.shape):
        if abs(matrix[ij]) < eps:
            matrix[ij] = 0
    return matrix


from ase.neighborlist import NeighborList


def shortest_vector_index(array):
    """
    Takes an array of vectors and finds the shortest one.
    
    Args:
    
        array: np.ndarray object
    
    Returns:
        int: Index of the shortest vector in the array.
    """
    idx = np.array([np.linalg.norm(vector) for vector in array]).argmin()
    return int(idx)


def gauss_reduce(vec1, vec2, tol=1e-6):
    """
    Get the shortest vectors in the lattice generated by
    the vectors vec1 and vec2 by using the Gauss reduction method.
    """
    reduced = False
    while not reduced:
        length1 = np.linalg.norm(vec1)
        length2 = np.linalg.norm(vec2)
        # First vector should be the shortest between the two
        if (length1 - length2) > tol:
            vec = vec1.copy()
            length = length1
            vec1 = vec2.copy()
            length1 = 1 * length2
            vec2 = vec.copy()
            length2 = 1 * length
        vec = vec2 - np.round(np.dot(vec1, vec2) / length1 ** 2) * vec1
        length = np.linalg.norm(vec)
        if length1 - length > tol:
            vec2 = vec1.copy()
            vec1 = vec.copy()
        else:
            vec2 = vec.copy()
            reduced = True
    return vec1, vec2


def check_neighbors(idx, neighbor_list, asecell, visited, layer):
    """
    Iterative function to get all atoms connected to the idx-th atom.
    
    Taken from https://github.com/epfl-theos/tool-layer-raman-ir/blob/master/compute/utils/layers.py.
    
    Args:
        idx (int): The index of the atom whose neighbors are checked.
        neighbor_list (ase.neighborlist.Neighborlist): The neighbor list object provided by ASE.
        asecell (ase.atoms.Atoms): The ASE structure.
        visited (list): The list of visited sites.
        layer (list): A list with atoms belonging to the current layer.
    """
    visited.append(idx)
    layer.append(idx)
    indeces, offsets = neighbor_list.get_neighbors(idx)
    for ref, offset in zip(indeces, offsets):
        if ref not in visited:
            if not all(offset == np.array([0, 0, 0])):
                asecell.positions[ref] += np.dot(offset, asecell.cell)
                neighbor_list.update(asecell)
            check_neighbors(ref, neighbor_list, asecell, visited, layer)


def _update_and_rotate_cell(asecell, newcell, layer_indices):
    """
    Update the cell according to the newcell provided, and then rotate it so that the first two lattice vectors are in the 
    x-y plane. Atomic positions are refolded moving each layer rigidly.
    
    Taken from https://github.com/epfl-theos/tool-layer-raman-ir/blob/master/compute/utils/layers.py.
    """
    asecell.set_cell(newcell)
    normal_vec = np.cross(newcell[0], newcell[1])
    asecell.rotate(v=normal_vec, a=[0, 0, 1], center=(0, 0, 0), rotate_cell=True)
    # it needs to be done twice because of possible bugs in ASE
    normal_vec = np.cross(asecell.cell[0], asecell.cell[1])
    asecell.rotate(v=normal_vec, a=[0, 0, 1], center=(0, 0, 0), rotate_cell=True)
    cell = asecell.cell
    # if the first two lattice vectors have equal magnitude and form
    # a 60deg angle, change the second so that the angle becomes 120
    if (abs(np.linalg.norm(cell[0]) - np.linalg.norm(cell[1])) < 1e-6) and (
        abs(np.dot(cell[0], cell[1]) / np.dot(cell[0], cell[0]) - 0.5) < 1e-3
    ):
        cell[1] -= cell[0]
    asecell.set_cell(cell)
    # finally rotate the first cell vector along x
    angle = np.arctan2(cell[0, 1], cell[0, 0]) * 180 / np.pi
    asecell.rotate(-angle, v=[0, 0, 1], center=(0, 0, 0), rotate_cell=True)
    # Wrap back in the unit cell each layer separately
    for layer in layer_indices:
        # projection of the atomic positions of the layer along the third axis
        proj = np.dot(asecell.positions[layer], [0, 0, 1])
        if len(layer_indices) == 1:
            # If there is only a single layer, center the atomic positions
            asecell.positions[layer] -= (
                proj.mean() / asecell.cell[2, 2] * asecell.cell[2]
            )
        else:
            # move back the vertical position of the layer within the cell
            asecell.positions[layer] -= (
                np.floor(proj.mean() / asecell.cell[2, 2]) * asecell.cell[2]
            )
    # fix also the inplane component of the positions
    asecell.pbc = [True, True, False]
    asecell.positions = asecell.get_positions(wrap=True)
    asecell.pbc = [True, True, True]
    return asecell


def find_layers(asecell, factor=1.1):
    """
    Obtains all subunits of a given structure by looking at the connectivity of the bonds.
    
    Taken from https://github.com/epfl-theos/tool-layer-raman-ir/blob/master/compute/utils/layers.py.

    Returns a tuple with a boolean indicating if the material is layered, a list of layers in the structure (ase format),
    a list of indices of the atoms in each layer, and a rotated bulk ASE cell (with stacking axis along z).
    MOREOVER, it 1) returns layers ordered by stacking index and 2) makes sure the layer is connected when 
    removing the PBC along the third (stacking) axis.
    
    Args:
        asecell (ase.atoms.Atoms): The bulk unit cell.
        factor (float): The skin factor of the neighborlist.
        
    Returns:
        tuple: (is_layered -> bool, list of sublayers -> list, indices of sublayers -> list)        
    """
    tol = 1.0e-6
    radii = factor * covalent_radii[asecell.get_atomic_numbers()]
    nl = NeighborList(radii, bothways=True, self_interaction=False, skin=0.0,)
    nl.update(asecell)
    vector1, vector2, vector3 = asecell.cell
    is_layered = True
    layer_structures = []
    layer_indices = []
    visited = []
    aselayer = None
    final_layered_structures = None

    # Loop over atoms (idx: atom index)
    for idx in range(len(asecell)):  # pylint: disable=too-many-nested-blocks
        # Will contain the indices of the atoms in the "current" layer
        layer = []
        # Check if I already visited this atom
        if idx not in visited:
            # Update 'layer' and 'visited'
            check_neighbors(idx, nl, asecell, visited, layer)
            aselayer = asecell.copy()[layer]
            subradii = factor * covalent_radii[aselayer.get_atomic_numbers()]
            layer_nl = NeighborList(
                subradii, bothways=True, self_interaction=False, skin=0.0,
            )
            layer_nl.update(aselayer)
            # We search for the periodic images of the first atom (idx=0)
            # that are connected to at least one atom of the connected layer
            neigh_vec = []
            for idx2 in range(len(aselayer)):
                _, offsets = layer_nl.get_neighbors(idx2)
                for offset in offsets:
                    if not all(offset == [0, 0, 0]):
                        neigh_vec.append(offset)
            # We define the dimensionality as the rank
            dim = np.linalg.matrix_rank(neigh_vec)
            if dim == 2:
                cell = asecell.cell
                vectors = list(np.dot(neigh_vec, cell))
                iv = shortest_vector_index(vectors)
                vector1 = vectors.pop(iv)
                iv = shortest_vector_index(vectors)
                vector2 = vectors.pop(iv)
                vector3 = np.cross(vector1, vector2)
                while np.linalg.norm(vector3) < tol:
                    iv = shortest_vector_index(vectors)
                    vector2 = vectors.pop(iv)
                    vector3 = np.cross(vector1, vector2)
                vector1, vector2 = gauss_reduce(vector1, vector2)
                vector3 = np.cross(vector1, vector2)
                aselayer = _update_and_rotate_cell(
                    aselayer, [vector1, vector2, vector3], [list(range(len(aselayer)))]
                )
                disconnected = []
                for i in range(-3, 4):
                    for j in range(-3, 4):
                        for k in range(-3, 4):
                            vector = i * cell[0] + j * cell[1] + k * cell[2]
                            if np.dot(vector3, vector) > tol:
                                disconnected.append(vector)
                iv = shortest_vector_index(disconnected)
                vector3 = disconnected[iv]
                layer_structures.append(aselayer)
                layer_indices.append(layer)
            else:
                is_layered = False
    if is_layered:
        newcell = [vector1, vector2, vector3]
        if abs(abs(np.linalg.det(newcell) / np.linalg.det(cell)) - 1.0) > 1e-3:
            raise ValueError(
                "An error occurred. The new cell after rotation has a different volume than the original cell"
            )
        rotated_asecell = _update_and_rotate_cell(asecell, newcell, layer_indices)
        # Re-order layers according to their projection
        # on the stacking direction
        vert_direction = np.cross(rotated_asecell.cell[0], rotated_asecell.cell[1])
        vert_direction /= np.linalg.norm(vert_direction)
        stack_proj = [
            np.dot(layer.positions, vert_direction).mean()
            for layer in [rotated_asecell[il] for il in layer_indices]
        ]
        stack_order = np.argsort(stack_proj)
        # order layers with increasing coordinate along the stacking direction
        layer_indices = [layer_indices[il] for il in stack_order]

        # Move the atoms along the third lattice vector so that
        # the first layer has zero projection along the vertical direction
        trans_vector = -(
            stack_proj[stack_order[0]]
            / np.dot(vert_direction, rotated_asecell.cell[2])
            * rotated_asecell.cell[2]
        )
        rotated_asecell.translate(trans_vector)

        # I don't return the 'layer_structures' because there the atoms are moved
        # from their positions and the z axis lenght might not be appropriate
        final_layered_structures = [
            rotated_asecell[this_layer_indices] for this_layer_indices in layer_indices
        ]
    else:
        rotated_asecell = None

    if not is_layered:
        aselayer = None
    return (
        is_layered,
        final_layered_structures,
        layer_indices,
    )

