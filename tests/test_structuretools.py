from aimstools.misc import *
import pytest

import ase.io
from pathlib import Path

from aimstools.structuretools import (
    Structure,
    find_fragments,
    find_periodic_axes,
    hexagonal_to_rectangular,
)


def test_structureclass():
    graphene = Path().cwd().joinpath("tests/structures/2d/graphene.xyz")
    try:
        strc = Structure(graphene)
    except Exception as inst:
        pytest.raises(inst)
    try:
        atoms = ase.io.read(graphene)
    except Exception as inst:
        pytest.raises(inst)
    strccopy = strc.copy()
    assert strccopy == strc, "Structure copy() method fails."
    a = strc.arrays
    b = atoms.arrays
    strcequal = (
        len(strc) == len(atoms)
        and (a["positions"] == b["positions"]).all()
        and (a["numbers"] == b["numbers"]).all()
        and (strc.cell == atoms.cell).all()
        and (strc.pbc == atoms.pbc).all()
    )
    assert strcequal == True, "Structure atoms inheritance fails."


def test_is_2D():
    testset = Path().cwd().joinpath("tests/structures/2d").glob("*.xyz")
    for xyz in testset:
        ML = Structure(xyz)
        assert True == ML.is_2d(), "Monolayer {} not recognised as 2d".format(
            str(xyz.parts[-1])
        )


def test_is_not_2d():
    testset = Path().cwd().joinpath("tests/structures/3d").glob("*.xyz")
    for xyz in testset:
        bulk = Structure(xyz)
        assert False == bulk.is_2d(), "Bulk {} falsely recognised as 2d.".format(
            str(xyz.parts[-1])
        )


def test_standard_with_enforced_axes():
    test = Structure(
        Path().cwd().joinpath("tests/structures/2d/MoS2_2H_1L_rotated.xyz")
    )
    try:
        test.standardize()
    except Exception as expt:
        pytest.raises(expt)
    assert (
        test.is_2d() == True
    ), "Standardization with permutation did not work correctly."


def test_hexagonal_to_rectangular():
    hex = Structure(Path().cwd().joinpath("tests/structures/2d/WS2_2H_1L.xyz"))
    rect = Structure(
        Path().cwd().joinpath("tests/structures/2d/WS2_2H_1L_rectangular.xyz")
    )
    hex_to_rect = hex.hexagonal_to_rectangular()
    a = rect.arrays
    b = hex_to_rect.arrays
    strcequal = (
        len(rect) == len(hex_to_rect)
        and (abs((a["positions"] - b["positions"])) < 1e-6).all()
        and (abs((a["numbers"] - b["numbers"])) < 1e-6).all()
        and (abs((rect.cell - hex_to_rect.cell)) < 1e-6).all()
        and (rect.pbc == hex_to_rect.pbc).all()
    )
    assert strcequal == True, "Conversion hexagonal to rectangular not working."
    hex_to_rect.standardize()
    assert (
        hex_to_rect.lattice == "hexagonal"
    ), "Back transformation to hexagonal not working."
