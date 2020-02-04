# Physical settings
bohr = 1.88973  # * Converts Angström to Bohr
hartree = 0.0367493  # * converts eV to Hartree

pse = {
    "H": 117,
    "He": 116,
    "Li": 115,
    "Be": 114,
    "B": 113,
    "C": 112,
    "N": 111,
    "O": 110,
    "F": 109,
    "Ne": 108,
    "Na": 107,
    "Mg": 106,
    "Al": 105,
    "Si": 104,
    "P": 103,
    "S": 102,
    "Cl": 101,
    "Ar": 100,
    "K": 99,
    "Ca": 98,
    "Sc": 97,
    "Ti": 96,
    "V": 95,
    "Cr": 94,
    "Mn": 93,
    "Fe": 92,
    "Co": 91,
    "Ni": 90,
    "Cu": 89,
    "Zn": 88,
    "Ga": 87,
    "Ge": 86,
    "As": 85,
    "Se": 84,
    "Br": 83,
    "Kr": 82,
    "Rb": 81,
    "Sr": 80,
    "Y": 79,
    "Zr": 78,
    "Nb": 77,
    "Mo": 76,
    "Tc": 75,
    "Ru": 74,
    "Rh": 73,
    "Pd": 72,
    "Ag": 71,
    "Cd": 70,
    "In": 69,
    "Sn": 68,
    "Sb": 67,
    "Te": 66,
    "I": 65,
    "Xe": 64,
    "Cs": 63,
    "Ba": 62,
    "La": 61,
    "Ce": 60,
    "Pr": 59,
    "Nd": 58,
    "Pm": 57,
    "Sm": 56,
    "Eu": 55,
    "Gd": 54,
    "Tb": 53,
    "Dy": 52,
    "Ho": 51,
    "Er": 50,
    "Tm": 49,
    "Yb": 48,
    "Lu": 47,
    "Hf": 46,
    "Ta": 45,
    "W": 44,
    "Re": 43,
    "Os": 42,
    "Ir": 41,
    "Pt": 40,
    "Au": 39,
    "Hg": 38,
    "Tl": 37,
    "Pb": 36,
    "Bi": 35,
    "Po": 34,
    "At": 33,
    "Rn": 32,
    "Fr": 31,
    "Ra": 30,
    "Ac": 29,
    "Th": 28,
    "Pa": 27,
    "U": 26,
    "Np": 25,
    "Pu": 24,
    "Am": 23,
    "Cm": 22,
    "Bk": 21,
    "Cf": 20,
    "Es": 19,
    "Fm": 18,
    "Md": 17,
    "No": 16,
    "Lr": 15,
    "Rf": 14,
    "Db": 13,
    "Sg": 12,
    "Bh": 11,
    "Hs": 10,
    "Mt": 9,
    "Ds": 8,
    "Rg": 7,
    "Cn": 6,
    "Nh": 5,
    "Fl": 4,
    "Mc": 3,
    "Lv": 2,
    "Ts": 1,
}

# Plotting specific settings
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TKAgg")
plt.style.use("seaborn-ticks")
plt.rcParams["legend.handlelength"] = 0.8
plt.rcParams["legend.framealpha"] = 0.8
font_name = "Arial"
font_size = 8.5
plt.rcParams.update({"font.sans-serif": font_name, "font.size": font_size})


# Logging settings
import logging
import time

logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)
