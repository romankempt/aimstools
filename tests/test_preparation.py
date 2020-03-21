from AIMS_tools.misc import *
import matplotlib.pyplot as plt
import time
import pytest
import shutil

import ase.io

from AIMS_tools import preparation


def test_prepare():
    if os.path.exists("tests/structures/tmp") == False:
        os.mkdir("tests/structures/tmp")
    shutil.copy("tests/structures/MoS2_ML.in", "tests/structures/tmp/MoS2_ML.in")
    prep = preparation.prepare(
        "tests/structures/tmp/MoS2_ML.in", **{"task": ["BS", "DOS", "GO", "phonons"]}
    )
    prep.setup_calculator()
    prep.adjust_control()
    prep.adjust_cost()
    prep.adjust_geometry()
    prep.write_submit_taurus()
    assert (
        os.path.exists("tests/structures/tmp/control.in") == True
    ), "control.in not written"
    assert (
        os.path.exists("tests/structures/tmp/MoS2_ML_BS_DOS_GO_phonons.sh") == True
    ), "submit script not written"
    shutil.rmtree("tests/structures/tmp", ignore_errors=True)
