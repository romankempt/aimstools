from aimstools.misc import *
from numpy import pi, sqrt


def monkhorstpack2kptdensity(atoms, k_grid):
    """Convert Monkhorst-Pack grid to k-point density.

    atoms (ase.atoms.Atoms): Atoms object.
    k_grid (list): [nx, ny, nz].

    Returns:
        float: Smallest line-density.
    """

    assert len(k_grid) == 3, "Size of k_grid is not 3."

    recipcell = atoms.cell.reciprocal()
    kd = []
    for i in range(3):
        if atoms.pbc[i]:
            kptdensity = k_grid[i] / (2 * pi * sqrt((recipcell[i] ** 2).sum()))
            kd.append(kptdensity)
    return round(min(kd), 2)

