from AIMS_tools.misc import *
import matplotlib.pyplot as plt
import time
import pytest
import shutil

import ase.io

from AIMS_tools import workflows
import tempfile


def test_kconv_prepare():
    dirpath = tempfile.mkdtemp()
    try:
        shutil.copy("tests/workflows/kconv/prep/MoS2.cif", dirpath)
        shutil.copy("tests/workflows/kconv/prep/Mo2S4.sh", dirpath)
    except:
        logging.error("Temporary directory not working.")
    kconv = workflows.k_convergence(
        os.path.join(dirpath, "MoS2.cif"),
        **{"submit": os.path.join(dirpath, "Mo2S4.sh")}
    )
    dirs = list(Path(dirpath).glob("*"))
    assert len(dirs) != 0, "kconv directories not created"
    shutil.rmtree(dirpath)


def test_kconv_evaluate():
    path = Path("tests/workflows/kconv/evaluate")
    workflows.k_convergence(path)
    shutil.move(path.joinpath("kconv.png"), "pictures/MoS2_kconv.png")


def test_kconv_testcases():
    path = Path("tests/workflows/kconv/testcases")
    for d in path.glob("*"):
        workflows.k_convergence(d)
        name = str(d.parts[-1])
        shutil.move(d.joinpath("kconv.png"), "pictures/{}_kconv.png".format(name))
