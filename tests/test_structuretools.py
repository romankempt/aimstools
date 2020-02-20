from AIMS_tools.misc import *
import matplotlib.pyplot as plt
import time
import pytest

import ase.io

from AIMS_tools import preparation, postprocessing, structuretools


def test_is_2D():
    ML = structuretools.structure("tests/structures/MoS2_ML.in")
    assert True == ML.is_2d(ML.atoms), "monolayer MoS2 not recognised"

    bulk = structuretools.structure("tests/structures/MoS2_bulk.in")
    assert False == bulk.is_2d(bulk.atoms), "bulk MoS2 not recognised"

    molecule = structuretools.structure("tests/structures/atom_in.xyz")
    assert False == molecule.is_2d(molecule.atoms), "molecule not recognised"

