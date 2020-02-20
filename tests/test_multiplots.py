from AIMS_tools.misc import *
import matplotlib.pyplot as plt
import time
import pytest

import ase.io

from AIMS_tools import (
    bandstructure,
    dos,
    phonons,
    multiplots,
    hirshfeld,
    postprocessing,
    eff_mass,
    preparation,
)


def test_bs_dos_combine():
    bs = bandstructure.bandstructure("tests/bandstructures/SOC_no_spin")
    ds = dos.density_of_states("tests/dos/NoSpin")
    fig = multiplots.combine(1, 2, [bs, ds], [4, 1])
    if fig.lines != None and not os.path.exists("pictures/MoS2_BS_DOS.png"):
        fig.suptitle("MoS$_2$ BS+DOS")
        plt.savefig("MoS2_BS_DOS.png", bbox_inches="tight", dpi=300)
        shutil.move("MoS2_BS_DOS.png", "pictures/")


def test_bs_dos_combine_both_spins():
    bs1 = bandstructure.bandstructure("tests/Spin/ZORA", spin="up")
    bs2 = bandstructure.bandstructure("tests/Spin/ZORA", spin="down")
    ds1 = dos.density_of_states("tests/Spin/ZORA", spin="up")
    ds2 = dos.density_of_states("tests/Spin/ZORA", spin="down")
    fig = multiplots.combine(1, 4, [bs1, ds1, bs2, ds2], [4, 1, 4, 1])
    if fig.lines != None and not os.path.exists("pictures/MnS2_ZORA_BS_DOS.png"):
        fig.suptitle("MnS$_2$ BS+DOS ZORA")
        plt.savefig("MnS2_ZORA_BS_DOS.png", bbox_inches="tight", dpi=300)
        shutil.move("MnS2_ZORA_BS_DOS.png", "pictures/")


# def test_bs_dos_combine_spin2():
#     bs = bandstructure.bandstructure("tests/Spin/SOC")
#     ds = dos.density_of_states("tests/Spin/SOC")
#     fig = multiplots.combine(1, 2, [bs, ds], [4, 1])
#     if fig.lines != None and not os.path.exists("pictures/MnS2_SOC_BS_DOS.png"):
#         fig.suptitle("MnS$_2$ BS+DOS SOC")
#         plt.savefig("MnS2_SOC_BS_DOS.png", bbox_inches="tight", dpi=300)
#         shutil.move("MnS2_SOC_BS_DOS.png", "pictures/")
