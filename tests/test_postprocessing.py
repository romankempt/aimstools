from aimstools.misc import *
import pytest

from pathlib import Path

from aimstools.postprocessing import FHIAimsOutputReader, HirshfeldReader


def test_output_reader_closed_shell():
    cs = Path().cwd().joinpath("tests/closed_shell")
    outr = FHIAimsOutputReader(cs)
    assert outr.is_converged, "Have a nice day not found."
    comp = {
        "xc": "pbe",
        "dispersion_correction": "MBD-nl",
        "relativistic": "atomic_zora scalar",
        "include_spin_orbit": True,
        "k_grid": (9, 9, 9),
        "spin": "none",
        "default_initial_moment": None,
        "fixed_spin_moment": None,
        "tasks": {
            "band structure",
            "mulliken-projected band structure",
            "total dos",
            "total dos tetrahedron",
            "atom-projected dos",
            "atom-projected dos tetrahedron",
            "species-projected dos",
            "species-projected dos tetrahedron",
        },
        "band_sections": [
            "output band 0.000000 0.000000 0.000000 \t 0.500000 0.000000 0.500000 \t 31 \t G X",
            "output band 0.500000 0.000000 0.500000 \t 0.500000 0.250000 0.750000 \t 31 \t X W",
            "output band 0.500000 0.250000 0.750000 \t 0.375000 0.375000 0.750000 \t 31 \t W K",
            "output band 0.375000 0.375000 0.750000 \t 0.000000 0.000000 0.000000 \t 31 \t K G",
            "output band 0.000000 0.000000 0.000000 \t 0.500000 0.500000 0.500000 \t 31 \t G L",
            "output band 0.500000 0.500000 0.500000 \t 0.625000 0.250000 0.625000 \t 31 \t L U",
            "output band 0.625000 0.250000 0.625000 \t 0.500000 0.250000 0.750000 \t 31 \t U W",
            "output band 0.500000 0.250000 0.750000 \t 0.500000 0.500000 0.500000 \t 31 \t W L",
            "output band 0.500000 0.500000 0.500000 \t 0.375000 0.375000 0.750000 \t 31 \t L K",
            "output band 0.625000 0.250000 0.625000 \t 0.500000 0.000000 0.500000 \t 31 \t U X",
        ],
        "mulliken_band_sections": [
            "output band_mulliken 0.000000 0.000000 0.000000 \t 0.500000 0.000000 0.500000 \t 31 \t G X",
            "output band_mulliken 0.500000 0.000000 0.500000 \t 0.500000 0.250000 0.750000 \t 31 \t X W",
            "output band_mulliken 0.500000 0.250000 0.750000 \t 0.375000 0.375000 0.750000 \t 31 \t W K",
            "output band_mulliken 0.375000 0.375000 0.750000 \t 0.000000 0.000000 0.000000 \t 31 \t K G",
            "output band_mulliken 0.000000 0.000000 0.000000 \t 0.500000 0.500000 0.500000 \t 31 \t G L",
            "output band_mulliken 0.500000 0.500000 0.500000 \t 0.625000 0.250000 0.625000 \t 31 \t L U",
            "output band_mulliken 0.625000 0.250000 0.625000 \t 0.500000 0.250000 0.750000 \t 31 \t U W",
            "output band_mulliken 0.500000 0.250000 0.750000 \t 0.500000 0.500000 0.500000 \t 31 \t W L",
            "output band_mulliken 0.500000 0.500000 0.500000 \t 0.375000 0.375000 0.750000 \t 31 \t L K",
            "output band_mulliken 0.625000 0.250000 0.625000 \t 0.500000 0.000000 0.500000 \t 31 \t U X",
        ],
        "qpe_calc": None,
        "use_dipole_correction": False,
    }
    for k in comp.keys():
        assert outr.control[k] == comp[k], "Key {} does not match {}.".format(k)
    outd = {
        "aims_version": "201103",
        "commit_number": "faf196098",
        "spin_N": 0,
        "spin_S": 0,
        "total_energy": -0.158031167777533e05,
        "band_extrema": (-5.62574054, -4.97520046, -5.61079465, -4.97519331),
        "fermi_level": (-5.58219687, -0.50198861e01, None, None),
        "work_function": None,
        "nkpoints": 365,
        "nscf_steps": 12,
    }
    for k in outd.keys():
        assert outr._outputdict[k] == outd[k], "Attribute {} does not match {}.".format(
            k
        )


def test_output_reader_open_shell():
    cs = Path().cwd().joinpath("tests/open_shell")
    outr = FHIAimsOutputReader(cs)
    assert outr.is_converged, "Have a nice day not found."
    comp = {
        "xc": "pbe",
        "dispersion_correction": "TS",
        "relativistic": "atomic_zora scalar",
        "include_spin_orbit": True,
        "k_grid": (6, 6, 6),
        "spin": "collinear",
        "default_initial_moment": 2.0,
        "fixed_spin_moment": None,
        "tasks": {
            "band structure",
            "atom-projected dos",
            "total dos",
            "mulliken-projected band structure",
            "species-projected dos",
        },
        "band_sections": [
            "output band 0.000000 0.000000 0.000000 \t 0.500000 -0.500000 0.500000 \t 31 \t G H",
            "output band 0.500000 -0.500000 0.500000 \t 0.000000 0.000000 0.500000 \t 31 \t H N",
            "output band 0.000000 0.000000 0.500000 \t 0.000000 0.000000 0.000000 \t 31 \t N G",
            "output band 0.000000 0.000000 0.000000 \t 0.250000 0.250000 0.250000 \t 31 \t G P",
            "output band 0.250000 0.250000 0.250000 \t 0.500000 -0.500000 0.500000 \t 31 \t P H",
            "output band 0.250000 0.250000 0.250000 \t 0.000000 0.000000 0.500000 \t 31 \t P N",
        ],
        "mulliken_band_sections": [
            "output band_mulliken 0.000000 0.000000 0.000000 \t 0.500000 -0.500000 0.500000 \t 31 \t G H",
            "output band_mulliken 0.500000 -0.500000 0.500000 \t 0.000000 0.000000 0.500000 \t 31 \t H N",
            "output band_mulliken 0.000000 0.000000 0.500000 \t 0.000000 0.000000 0.000000 \t 31 \t N G",
            "output band_mulliken 0.000000 0.000000 0.000000 \t 0.250000 0.250000 0.250000 \t 31 \t G P",
            "output band_mulliken 0.250000 0.250000 0.250000 \t 0.500000 -0.500000 0.500000 \t 31 \t P H",
            "output band_mulliken 0.250000 0.250000 0.250000 \t 0.000000 0.000000 0.500000 \t 31 \t P N",
        ],
        "qpe_calc": None,
        "use_dipole_correction": False,
    }
    for k in comp.keys():
        assert outr.control[k] == comp[k], "Key {} does not match {}.".format(k)
    outd = {
        "aims_version": "201103",
        "commit_number": "faf196098",
        "spin_N": 2.22204,
        "spin_S": 1.11102,
        "total_energy": -34767.8892822522,
        "band_extrema": (-9.16890066, -8.84338083, -8.87712372, -8.84392225),
        "fermi_level": (-8.87479742, -8.8757381, None, None),
        "work_function": None,
        "nkpoints": 216,
        "nscf_steps": 15,
    }
    for k in outd.keys():
        assert outr._outputdict[k] == outd[k], "Attribute {} does not match {}.".format(
            k
        )


def test_output_reader_open_shell_fixed():
    cs = Path().cwd().joinpath("tests/open_shell_fixed_moment")
    outr = FHIAimsOutputReader(cs)
    assert outr.is_converged, "Have a nice day not found."
    comp = {
        "xc": "pbe",
        "dispersion_correction": None,
        "relativistic": "atomic_zora scalar",
        "include_spin_orbit": True,
        "k_grid": (6, 6, 6),
        "spin": "collinear",
        "default_initial_moment": 2.0,
        "fixed_spin_moment": 2.0,
        "tasks": {
            "band structure",
            "atom-projected dos",
            "mulliken-projected band structure",
        },
        "band_sections": [
            "output band 0.000000 0.000000 0.000000 \t 0.500000 -0.500000 0.500000 \t 31 \t G H",
            "output band 0.500000 -0.500000 0.500000 \t 0.000000 0.000000 0.500000 \t 31 \t H N",
            "output band 0.000000 0.000000 0.500000 \t 0.000000 0.000000 0.000000 \t 31 \t N G",
            "output band 0.000000 0.000000 0.000000 \t 0.250000 0.250000 0.250000 \t 31 \t G P",
            "output band 0.250000 0.250000 0.250000 \t 0.500000 -0.500000 0.500000 \t 31 \t P H",
            "output band 0.250000 0.250000 0.250000 \t 0.000000 0.000000 0.500000 \t 31 \t P N",
        ],
        "mulliken_band_sections": [
            "output band_mulliken 0.000000 0.000000 0.000000 \t 0.500000 -0.500000 0.500000 \t 31 \t G H",
            "output band_mulliken 0.500000 -0.500000 0.500000 \t 0.000000 0.000000 0.500000 \t 31 \t H N",
            "output band_mulliken 0.000000 0.000000 0.500000 \t 0.000000 0.000000 0.000000 \t 31 \t N G",
            "output band_mulliken 0.000000 0.000000 0.000000 \t 0.250000 0.250000 0.250000 \t 31 \t G P",
            "output band_mulliken 0.250000 0.250000 0.250000 \t 0.500000 -0.500000 0.500000 \t 31 \t P H",
            "output band_mulliken 0.250000 0.250000 0.250000 \t 0.000000 0.000000 0.500000 \t 31 \t P N",
        ],
        "qpe_calc": None,
        "use_dipole_correction": False,
    }

    for k in comp.keys():
        assert outr.control[k] == comp[k], "Key {} does not match {}.".format(k)
    outd = {
        "aims_version": "201103",
        "commit_number": "faf196098",
        "spin_N": 2.0,
        "spin_S": 1.0,
        "total_energy": -34767.0342962745,
        "band_extrema": (-8.60761003, -9.08189184, -9.07974406, -8.83731232),
        "fermi_level": (
            None,
            -8.9161675,
            -9.1563046488,
            -8.5621614354,
        ),
        "work_function": None,
        "nkpoints": 216,
        "nscf_steps": 13,
    }
    for k in outd.keys():
        assert outr._outputdict[k] == outd[k], "Attribute {} does not match {}.".format(
            k
        )


def test_output_reader_work_function():
    cs = Path().cwd().joinpath("tests/work_function")
    outr = FHIAimsOutputReader(cs)
    assert outr.is_converged, "Have a nice day not found."
    comp = {
        "xc": "pbe",
        "dispersion_correction": None,
        "relativistic": "atomic_zora scalar",
        "include_spin_orbit": True,
        "k_grid": (6, 6, 1),
        "spin": "none",
        "default_initial_moment": None,
        "fixed_spin_moment": None,
        "tasks": {
            "band structure",
            "atom-projected dos",
            "mulliken-projected band structure",
        },
        "band_sections": [
            "output band 0.000000 0.000000 0.000000 \t 0.500000 0.000000 0.000000 \t 31 \t G M",
            "output band 0.500000 0.000000 0.000000 \t 0.333333 0.333333 0.000000 \t 31 \t M K",
            "output band 0.333333 0.333333 0.000000 \t 0.000000 0.000000 0.000000 \t 31 \t K G",
        ],
        "mulliken_band_sections": [
            "output band_mulliken 0.000000 0.000000 0.000000 \t 0.500000 0.000000 0.000000 \t 31 \t G M",
            "output band_mulliken 0.500000 0.000000 0.000000 \t 0.333333 0.333333 0.000000 \t 31 \t M K",
            "output band_mulliken 0.333333 0.333333 0.000000 \t 0.000000 0.000000 0.000000 \t 31 \t K G",
        ],
        "qpe_calc": None,
        "use_dipole_correction": True,
    }

    for k in comp.keys():
        assert outr.control[k] == comp[k], "Key {} does not match {}.".format(k)
    outd = {
        "aims_version": "200819",
        "commit_number": "009f8e893",
        "spin_N": 0,
        "spin_S": 0,
        "total_energy": -2169.26098443649,
        "band_extrema": (
            -6.04870044,
            -1.40809218,
            -6.04869548,
            -1.40810407,
        ),
        "fermi_level": (-2.85009039, -2.7033213, None, None),
        "work_function": (
            -0.01324087,
            -0.01324087,
            2.83684953,
            2.83684953,
        ),
        "nkpoints": 20,
        "nscf_steps": 11,
    }
    for k in outd.keys():
        assert outr._outputdict[k] == outd[k], "Attribute {} does not match {}.".format(
            k
        )


def test_hirshfeld_reader():
    cs = Path().cwd().joinpath("tests/hirshfeld_charges")
    hfr = HirshfeldReader(cs)
    assert hfr.is_converged, "Have a nice day not found."
    assert (
        "hirshfeld charge analysis" in hfr.control["tasks"]
    ), "Charge analysis not found in control.in!"

    c1 = {(0, "N"): -0.19904565, (1, "B"): 0.19898187}
    c2 = hfr.charges
    for k1, v1 in c1.items():
        assert c2[k1] == v1, "Charge reading did not work correctly."
    c1 = {"N": -0.19904565, "B": 0.19898187}
    c2 = hfr.total_charges
    for k1, v1 in c1.items():
        assert c2[k1] == v1, "Total charge reading did not work correctly."
