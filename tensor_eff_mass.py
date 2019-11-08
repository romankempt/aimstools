import numpy as np
import glob, sys, os, shutil
import matplotlib.pyplot as plt
from pathlib import Path as Path
from scipy import interpolate
from AIMS_tools import bandstructure
import ase.io, ase.cell, ase.spacegroup
import math
from scipy.optimize import curve_fit
from mpl_toolkits import mplot3d
from matplotlib import cm


# os.chdir(
#     r"AIMS_tools\Tests\Silicon\PostSCF"
# )
hartree_to_eV = 27.211402666173235  # eV to hartree
Angstroem_to_bohr = 1.889725989


def get_output_information(outputfile):
    """ Retrieve information such as Fermi level 
    and band gap from output file."""
    with open(outputfile, "r") as file:
        for line in file:
            if "Chemical potential (Fermi level) in eV" in line:
                return float(line.split()[-1])


def read_geometry(geometryfile):
    """ Retrieve atom types and number of atoms from geometry. """
    cell = ase.io.read(geometryfile)
    rec_cell = cell.get_reciprocal_cell() * 2 * math.pi / Angstroem_to_bohr
    rec_cell_lengths = ase.cell.Cell.new(
        rec_cell
    ).lengths()  # converting to atomic units 2 pi/bohr
    return rec_cell_lengths


fermi_level = get_output_information("Si.out")
rec_cell_lengths = read_geometry("geometry.in")

with open("Final_KS_eigenvalues.dat", "r") as file:
    content = [
        i.split()
        for i in file.readlines()
        if "k-point" not in i and "#" not in i and i != ""
    ]
    content = list(filter(None, content))
    content = np.array(content, dtype=float)
    state, occupation, energy = content[:, 0], content[:, 1], content[:, 2]
with open("Final_KS_eigenvalues.dat", "r") as file:
    kpoints = [
        i.split()[-3:]
        for i in file.readlines()
        if "k-point in recip. lattice units:" in i
    ]
    kpoints = np.array(kpoints, dtype=float)

rows = len(kpoints)
cols = int(np.max(state))
energy = (energy.reshape(rows, cols) - fermi_level) / hartree_to_eV
occupation = occupation.reshape(rows, cols)


# # array of  x y z coordinate and then eigenvalue for every state
# this array is TRS reduced!
# Steps: Fully recover spectrum without TRS
kpoints = np.vstack((-kpoints[::-1], kpoints))
energy = np.vstack((energy[::-1], energy))
spectrum = np.hstack((kpoints, energy))
print(spectrum.shape)
u, indices = np.unique(spectrum, axis=0, return_inverse=True)
spectrum = u
print(spectrum.shape)
# print(spectrum.shape)
# work in reciprocal coordinates
# find minima
# shift to minima
# find all points in a radius around minimum, including the ones
# on the other side of the BZ


VBM = np.max(spectrum[:, 3:][spectrum[:, 3:] < 0])
index = np.where(spectrum[:, 3:] == VBM)
VBMloc = spectrum[index[0][0], 0:3]
band = index[1][0]

band = spectrum[:, [0, 1, 2, band]]
band[:, [0, 1, 2]] -= VBMloc  # shift to location
cutoff_radius = 0.05
dist = np.zeros(spectrum.shape[0])
for i in range(len(dist)):
    dist[i] = np.linalg.norm(band[i, [0, 1, 2]])
band = band[dist < cutoff_radius]
band[:, [0, 1, 2]] *= rec_cell_lengths


def tensor(Vars, E0, mxx, myy, mzz, mxy, mxz, myz):
    x, y, z = Vars[:, 0], Vars[:, 1], Vars[:, 2]
    return (
        E0
        + 1 / (2 * mxx) * x * x
        + 1 / (2 * myy) * y * y
        + 1 / (2 * mzz) * z * z
        + 1 / (2 * mxy) * x * y
        + 1 / (2 * mxz) * x * z
        + 1 / (2 * myz) * y * z
    )


p0 = VBM, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1
bounds = (
    (VBM - 0.00001, -10, -10, -10, -10, -10, -10),
    (VBM + 0.00001, 10, 10, 10, 10, 10, 10),
)
popt, pcov = curve_fit(tensor, band[:, 0:3], band[:, 3], bounds=bounds)
err = np.sqrt(np.diag(pcov))[1]

mtens = np.array(
    [
        [popt[1], popt[4], popt[5]],
        [popt[4], popt[2], popt[6]],
        [popt[5], popt[6], popt[3]],
    ]
)

w, v = np.linalg.eig(mtens)
