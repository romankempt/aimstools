from AIMS_tools.misc import *
import matplotlib.pyplot as plt
import time
import pytest
import shutil

import ase.io

from AIMS_tools import preparation

import tempfile


def test_prepare():
    dirpath = tempfile.mkdtemp()
    shutil.copy("tests/structures/MoS2_ML.in", dirpath)
    prep = preparation.prepare(
        os.path.join(dirpath, "MoS2_ML.in"), **{"task": ["BS", "DOS", "GO", "phonons"]}
    )
    prep.setup_calculator()
    prep.adjust_control()
    prep.adjust_cost()
    prep.adjust_geometry()
    prep.write_submit_taurus()
    shutil.rmtree(dirpath)
