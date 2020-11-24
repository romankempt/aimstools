from aimstools.misc import *
import pytest

from pathlib import Path

import tempfile
import shutil

from aimstools.preparation import FHIAimsSetup, FHIVibesSetup
import ase.io


def test_aims_setup():
    dirpath = tempfile.mkdtemp()
    shutil.copy("tests/preparation/hBN.xyz", dirpath)
    geometryfile = Path(dirpath).joinpath("hBN.xyz")
    ais = FHIAimsSetup(geometryfile, tasks=["BS", "fatBS", "dos"])
    assert ais.tasks == {
        "band structure",
        "mulliken-projected band structure",
        "total dos tetrahedron",
        "atom-projected dos tetrahedron",
    }, "Tasks not parsed correctly."

    ais.setup_geometry()
    georef = ase.io.read("tests/preparation/geometry.in")
    geocomp = ase.io.read(Path(dirpath).joinpath("geometry.in"))
    assert georef == geocomp, "File geometry.in not written correctly."

    ais.setup_control()
    controlref = Path("tests/preparation/control.in")
    controlcomp = Path(dirpath).joinpath("control.in")
    with open(controlref, "r") as f1, open(controlcomp, "r") as f2:
        s1 = [line.strip() for line in f1.readlines() if not line.startswith("#")]
        s2 = [line.strip() for line in f2.readlines() if not line.startswith("#")]

    for l in s1:
        assert l in s2, "Line {} not found in control.in.".format(l)
    shutil.rmtree(dirpath)


def test_vibes_setup_relaxation():
    dirpath = tempfile.mkdtemp()
    shutil.copy("tests/preparation/hBN.xyz", dirpath)
    geometryfile = Path(dirpath).joinpath("hBN.xyz")
    vibup = FHIVibesSetup(geometryfile, tasks=["relaxation"])
    assert vibup.tasks == {"relaxation"}, "Tasks not parsed correctly."

    vibup.setup_relaxation()
    ref = Path("tests/preparation/relaxation.in")
    comp = Path(dirpath).joinpath("relaxation.in")
    with open(ref, "r") as f1, open(comp, "r") as f2:
        s1 = [line for line in f1.readlines() if not line.startswith("#")]
        s2 = [line for line in f2.readlines() if not line.startswith("#")]
    for l in s1:
        assert l in s2, "Line {} not found in control.in.".format(l)
    shutil.rmtree(dirpath)


def test_vibes_setup_phonopy():
    dirpath = tempfile.mkdtemp()
    shutil.copy("tests/preparation/hBN.xyz", dirpath)
    geometryfile = Path(dirpath).joinpath("hBN.xyz")
    vibup = FHIVibesSetup(geometryfile, tasks=["phonons"])
    assert vibup.tasks == {"phonons"}, "Tasks not parsed correctly."

    vibup.setup_phonopy()
    ref = Path("tests/preparation/phonopy.in")
    comp = Path(dirpath).joinpath("phonopy.in")
    with open(ref, "r") as f1, open(comp, "r") as f2:
        s1 = [line for line in f1.readlines() if not line.startswith("#")]
        s2 = [line for line in f2.readlines() if not line.startswith("#")]
    for l in s1:
        assert l in s2, "Line {} not found in control.in.".format(l)
    shutil.rmtree(dirpath)
