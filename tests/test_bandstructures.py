from AIMS_tools.misc import *
import matplotlib.pyplot as plt
import time
import pytest

import ase.io

from AIMS_tools import (
    bandstructure,
    dos,
    multiplots,
)


def test_zora_bandstructure():
    bs = bandstructure.bandstructure("tests/bandstructures/ZORA_no_spin")
    assert bs.structure.atoms == ase.io.read(
        "tests/bandstructures/ZORA_no_spin/geometry.in"
    ), " zora geometry not correctly read"
    assert bs.kpath == ["G", "M", "K", "G"], "zora kpath reading not working"
    bs.custom_path("G-M-K-M-G")
    assert bs.kpath == ["G", "M", "K", "M", "G"], "zora custom path not working"
    p1 = bs.plot()
    if p1.lines != None and not os.path.exists("pictures/MoS2_ZORA_BS.png"):
        p1.set_title("MoS$_2$ ZORA")
        plt.savefig("MoS2_ZORA_BS.png", bbox_inches="tight", dpi=300)
        shutil.move("MoS2_ZORA_BS.png", "pictures/")


def test_soc_bandstructure():
    bs = bandstructure.bandstructure("tests/bandstructures/SOC_no_spin")
    assert bs.structure.atoms == ase.io.read(
        "tests/bandstructures/SOC_no_spin/geometry.in"
    ), " soc geometry not correctly read"
    assert bs.kpath == ["G", "M", "K", "G"], "soc kpath reading not working"
    bs.custom_path("G-M-K-M-G")
    assert bs.kpath == ["G", "M", "K", "M", "G"], "soc custom path not working"
    p1 = bs.plot()
    if p1.lines != None and not os.path.exists("pictures/MoS2_soc_BS.png"):
        p1.set_title("MoS$_2$ SOC")
        plt.savefig("MoS2_SOC_BS.png", bbox_inches="tight", dpi=300)
        shutil.move("MoS2_SOC_BS.png", "pictures/")


def test_zora_fatbandstructure():
    bs = bandstructure.fatbandstructure("tests/fatbandstructures/ZORA")
    p1 = bs.plot_all_species()
    if p1.lines != None and not os.path.exists("pictures/MoSe2_ZORA_fatBS.png"):
        p1.set_title("MoSe$_2$ ZORA fat BS")
        plt.tight_layout()
        plt.savefig("MoSe2_ZORA_fatBS.png", dpi=300)
        shutil.move("MoSe2_ZORA_fatBS.png", "pictures/")
    p2 = bs.plot_all_orbitals()
    if p2.lines != None and not os.path.exists("pictures/MoSe2_ZORA_fatBS_orbs.png"):
        p2.set_title("MoSe$_2$ ZORA fat BS orbs")
        plt.savefig("MoSe2_ZORA_fatBS_orbs.png", bbox_inches="tight", dpi=300)
        shutil.move("MoSe2_ZORA_fatBS_orbs.png", "pictures/")


def test_fatbs_read_and_write():
    if Path("tests/fatbandstructures/SOC/fatbands_atom_contributions.npz").exists():
        os.remove(Path("tests/fatbandstructures/SOC/fatbands_atom_contributions.npz"))
    bs = bandstructure.fatbandstructure("tests/fatbandstructures/SOC")
    bs2 = bandstructure.fatbandstructure("tests/fatbandstructures/SOC")
    for index, atom in bs.atoms_to_plot.items():
        for section in bs.mlk_bandsegments.keys():
            k1, ev1 = bs.atom_contributions[index][section]
            k2, ev2 = bs2.atom_contributions[index][section]
            assert np.array_equal(k1, k2), "kaxis not matching"
            assert np.array_equal(ev1, ev2), "eigenvalues not matching"

