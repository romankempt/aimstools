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
        atoms: :class:`~ase.atoms.Atoms` or :class:`~aimstools.structuretools.structure.Structure`

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


def find_nonperiodic_axes(atoms) -> dict:
    """Evaluates if given structure is qualitatively periodic along certain lattice directions.

    Args:
        atoms: ase.atoms.Atoms object.

    Note:
        A criterion is a vacuum space of more than 25.0 Anström.

    Returns:
        dict: Axis : Bool pairs.
    """
    atoms = atoms.copy()
    sc = ase.build.make_supercell(atoms, 2 * np.identity(3), wrap=True)
    fragments = find_fragments(sc)
    crit1 = True if len(fragments) > 1 else False
    logger.debug("Len fragments: {:d}".format(len(fragments)))
    pbc = dict(zip([0, 1, 2], [True, True, True]))
    logger.debug("sc cell lengths: {} {} {}".format(*sc.cell.lengths()))
    if crit1:
        for axes in (0, 1, 2):
            spans = []
            for tup in fragments:
                start = np.min(tup.atoms.get_positions()[:, axes])
                end = np.max(tup.atoms.get_positions()[:, axes])
                spans.append((start, end))
            spans = list(set(spans))
            spans = sorted(spans, key=lambda x: x[0])
            logger.debug("Len spans: {:d}".format(len(spans)))
            if len(spans) > 1:
                for k, l in zip(spans[:-1], spans[1:]):
                    d1 = abs(k[1] - l[0])
                    d2 = abs(
                        k[1] - l[0] - sc.cell.lengths()[axes]
                    )  # check if fragments are separated by a simple translation
                    nd = np.min([d1, d2])
                    logger.debug("Axes: {}, nd: {}".format(axes, nd))
                    if nd >= 25.0:
                        pbc[axes] = False
                        break
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
