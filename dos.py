# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 17:25:27 2019

@author: Roman Kempt
"""

import numpy as np
import glob, sys, os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
import re
from pathlib import Path as Path
import argparse


class DOS:
    """ Contains all DOS information. """

    def __init__(self, outputfile, get_SOC=True):
        """
        outputfile :  str (path to DOS outputfile)
        get_SOC : True or False to evaluate SOC data.
        """
        cwd = Path.cwd()
        self.path = cwd.joinpath(
            Path(outputfile).parent
        )  # this retrieves the directory where the output file is
        self.DOS_type, self.active_SOC = self.__read_control()
        self.VBM, self.CBM, self.fermi_level = self.__read_output(outputfile)
        self.band_gap = self.CBM - self.VBM
        self.species = self.__read_geometry()

        dosfiles = self.__get_raw_data(self.active_SOC, get_SOC)
        dos_per_atom = []
        for atom in self.species.keys():
            atomic_dos = self.__sort_dosfiles(dosfiles, atom)
            atomic_dos = self.__sum_dosfiles(atomic_dos)
            dos_per_atom.append(atomic_dos)
        self.dos_per_atom = dict(zip(self.species.keys(), dos_per_atom))
        for atom in self.dos_per_atom.keys():
            self.dos_per_atom[atom][:, 1:] = (
                self.dos_per_atom[atom][:, 1:] * self.species[atom]
            )
        self.shift_to("middle")

        self.total_dos = self.__get_total_dos(self.dos_per_atom)

    def __read_control(self):
        """ Retrieve DOS and SOC information. """
        control = self.path.joinpath("control.in")
        with open(control, "r") as file:
            for line in file.readlines():
                if "output atom_proj_dos" in line:
                    DOS_type = "atom_proj_dos"
                if "include_spin_orbit" in line:
                    active_SOC = True
        return DOS_type, active_SOC

    def __read_geometry(self):
        """ Retrieve atom types and number of atoms from geometry. """
        geometry = self.path.joinpath("geometry.in")
        atoms = []
        with open(geometry, "r") as file:
            for line in file.readlines():
                if "atom" in line:
                    atoms.append(line.split()[-1])
        keys = list(set(atoms))
        numbers = []
        for key in keys:
            numbers.append(atoms.count(key))
        species = dict(zip(keys, numbers))
        return species

    def __get_raw_data(self, active_SOC, get_SOC):
        """ Get .raw.out files.
            get_SOC : True or False to obtain the ones with or without SOC.
            """
        if self.active_SOC == True:
            if get_SOC == True:
                dosfiles = list(self.path.glob("*raw*"))
                dosfiles = [str(i) for i in dosfiles if "no_soc" not in str(i)]
            elif get_SOC == False:
                dosfiles = list(self.path.glob("*raw*"))
                dosfiles = [str(i) for i in dosfiles if "no_soc" in str(i)]
        else:
            dosfiles = list(self.path.glob("*raw*"))
        return dosfiles

    def __sort_dosfiles(self, dosfiles, atom_type):
        """ Find the dosfiles that match the atom name + four numbers."""
        pattern = re.compile(atom_type + r"\d{4}")
        atom_wise_files = list(filter(pattern.search, dosfiles))
        return atom_wise_files

    def __sum_dosfiles(self, atom_wise_files):
        """ Dosfiles contain the energy axes, the total DOS, and then
        the DOS per angular momentum l=0, l=1 ... """
        list_of_arrays = []
        for entry in atom_wise_files:
            array = np.loadtxt(entry, dtype=float, comments="#")
            list_of_arrays.append(array)
        array = np.sum(list_of_arrays, axis=0) / len(list_of_arrays)
        return array

    def __get_total_dos(self, dos_per_atom):
        """ Sum over the total dos column of all atom DOS files."""
        total_dos = []
        for atom in dos_per_atom.keys():
            total_dos.append(dos_per_atom[atom][:, [0, 1]])
        total_dos = np.sum(total_dos, axis=0) / len(dos_per_atom.keys())
        return total_dos

    def __read_output(self, outputfile):
        """ Retrieve VBM, CBM and Fermi level from output file. """
        with open(outputfile, "r") as file:
            for line in file.readlines():
                if "Highest occupied state (VBM) at" in line:
                    VBM = float(line.split()[5])
                if "Lowest unoccupied state (CBM) at" in line:
                    CBM = float(line.split()[5])
                if "Chemical potential (Fermi level) in eV" in line:
                    fermi_level = float(line.split()[-1])
        return VBM, CBM, fermi_level

    def shift_to(self, shift_type):
        """ Shifts Fermi level.
                
            shift_type = "middle" to shift to middle of band gap
            shift_type = "VBM" to shift to valence band maximum"
            shift_type = None to add internal Fermi-level. This is the default for metallic systems."""
        if self.band_gap < 0.1:
            shift_type = None
            for atom in self.dos_per_atom.keys():
                self.dos_per_atom[atom][:, 0] -= self.fermi_level
        elif shift_type == "middle":
            for atom in self.dos_per_atom.keys():
                self.dos_per_atom[atom][:, 0] -= (self.VBM + self.CBM) / 2
        elif shift_type == "VBM":
            for atom in self.dos_per_atom.keys():
                self.dos_per_atom[atom][:, 0] -= self.VBM

    def plot_single_atomic_dos(
        self,
        atom,
        color,
        orbital=None,
        fig=None,
        axes=None,
        title="",
        fill="gradient",
        var_energy_limits=1.0,
        fix_energy_limits=[],
        kwargs={},
    ):
        """ Plot total DOS of input species.
            ToDo: write angular momentum dependency

            var_energy_limits = 1.0 are variable y-limits, which scale y above and below band gap.
            fix_energy_limits is a list of fixed y-limits.
            fill : "gradient", "constant", or None
            orbital:    None --> Total DOS
                        s    --> l = 0
                        p    --> l = 1
                        d    --> l = 2
                        f    --> l = 3 etc.
            **kwargs are passed to matplotlib plotting function.
            Returns axes object."""
        orbitals = {None: "", "s": 2, "p": 3, "d": 4, "f": 5}
        if fig == None:
            fig = plt.figure(figsize=(2, 4))
        if axes == None:
            axes = plt.gca()
        if orbital == None:  # This retrieves the total DOS.
            xy = self.dos_per_atom[atom][:, [0, 1]]
        else:
            xy = self.dos_per_atom[atom][:, [0, orbitals[orbital]]]

        if fix_energy_limits == []:
            lower_ylimit = -self.band_gap - var_energy_limits
            upper_ylimit = self.band_gap + var_energy_limits
        else:
            lower_ylimit = fix_energy_limits[0]
            upper_ylimit = fix_energy_limits[1]

        ### The y-range is cut out for two reasons: Scaling the plotted range
        ### and having the gradients with better colors.
        xy = xy[(xy[:, 0] > lower_ylimit - 3) & (xy[:, 0] < upper_ylimit + 3)]
        x = xy[:, 1]
        y = xy[:, 0]

        if fill == None:
            axes.plot(x, y, color=color, **kwargs)
        elif fill == "gradient":
            axes.plot(x, y, color=color, alpha=0.9)
            self.__gradient_fill(x, y, axes, color)
        elif fill == "constant":
            axes.fill_betweenx(y, 0, x, color=color)

        xy = xy[(xy[:, 0] > lower_ylimit) & (xy[:, 0] < upper_ylimit)]
        axes.set_xlim([0, np.max(xy[:, 1]) + 0.1])
        axes.set_ylim([lower_ylimit, upper_ylimit])
        axes.set_xticks([])
        axes.set_xlabel("DOS")
        axes.set_ylabel("E-E$_\mathrm{F}$ [eV]")
        axes.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axes.axhline(y=0, color="k", alpha=0.5, linestyle="--")
        axes.set_title(str(title), loc="center")
        return axes

    def __gradient_fill(self, x, y, axes, color):
        """
        Plot a linear alpha gradient beneath x y values.
        Here, x and y are transposed due to the nature of DOS graphs.
    
        Parameters
        ----------
        x, y : array-like
            The data values of the line.
        Additional arguments are passed on to matplotlib's ``plot`` function.
        """
        z = np.empty((1, 100, 4), dtype=float)
        rgb = mcolors.colorConverter.to_rgb(color)
        z[:, :, :3] = rgb
        z[:, :, -1] = np.linspace(0, 0.9, 100)[None, :]
        _, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
        im = axes.imshow(z, aspect="auto", extent=[0, xmax, ymin, ymax], origin="upper")
        xy = np.column_stack([x, y])
        xy = np.vstack([[ymin, 0], xy, [ymin, xmax], [ymin, 0]])
        clip_path = Polygon(xy, facecolor="none", edgecolor="none", closed=True)
        axes.add_patch(clip_path)
        im.set_clip_path(clip_path)

    def plot_all_atomic_dos(
        self,
        fig=None,
        axes=None,
        var_energy_limits=1.0,
        fix_energy_limits=[],
        kwargs={},
    ):
        """ In-built function to get DOS plot of all atomic species
        with default JMOL colors. """
        if fig == None:
            fig = plt.figure(figsize=(2, 4))
        if axes != None:
            axes = plt.gca()
        else:
            axes = plt.subplot2grid((1, 1), (0, 0), fig=fig)
        handles = []
        xmax = []
        color_dict["P"] = "red"
        color_dict["Pt"] = "royalblue"
        atoms = list(self.species.keys())
        atoms.sort()
        for atom in atoms:
            if atom == "H":
                continue
            color = color_dict[atom]
            axes = self.plot_single_atomic_dos(
                atom,
                color,
                var_energy_limits=var_energy_limits,
                fix_energy_limits=fix_energy_limits,
                fig=fig,
                axes=axes,
                kwargs=kwargs,
            )
            xmax.append(axes.get_xlim()[1])
            handles.append(Line2D([0], [0], color=color, label=atom, lw=1.0))
        axes.legend(handles=handles, frameon=True, loc="center right", fancybox=False)
        axes.set_xlim([0, max(xmax)])
        return axes


#########################################
color_dict = {
    "H": (1, 1, 1),
    "He": (0.850980392, 1, 1),
    "Li": (0.8, 0.501960784, 1),
    "Be": (0.760784314, 1, 0),
    "B": (1, 0.709803922, 0.709803922),
    "C": (0.564705882, 0.564705882, 0.564705882),
    "N": (0.188235294, 0.31372549, 0.97254902),
    "O": (1, 0.050980392, 0.050980392),
    "F": (0.564705882, 0.878431373, 0.31372549),
    "Ne": (0.701960784, 0.890196078, 0.960784314),
    "Na": (0.670588235, 0.360784314, 0.949019608),
    "Mg": (0.541176471, 1, 0),
    "Al": (0.749019608, 0.650980392, 0.650980392),
    "Si": (0.941176471, 0.784313725, 0.62745098),
    "P": (1, 0.501960784, 0),
    "S": (1, 1, 0.188235294),
    "Cl": (0.121568627, 0.941176471, 0.121568627),
    "Ar": (0.501960784, 0.819607843, 0.890196078),
    "K": (0.560784314, 0.250980392, 0.831372549),
    "Ca": (0.239215686, 1, 0),
    "Sc": (0.901960784, 0.901960784, 0.901960784),
    "Ti": (0.749019608, 0.760784314, 0.780392157),
    "V": (0.650980392, 0.650980392, 0.670588235),
    "Cr": (0.541176471, 0.6, 0.780392157),
    "Mn": (0.611764706, 0.478431373, 0.780392157),
    "Fe": (0.878431373, 0.4, 0.2),
    "Co": (0.941176471, 0.564705882, 0.62745098),
    "Ni": (0.31372549, 0.815686275, 0.31372549),
    "Cu": (0.784313725, 0.501960784, 0.2),
    "Zn": (0.490196078, 0.501960784, 0.690196078),
    "Ga": (0.760784314, 0.560784314, 0.560784314),
    "Ge": (0.4, 0.560784314, 0.560784314),
    "As": (0.741176471, 0.501960784, 0.890196078),
    "Se": (1, 0.631372549, 0),
    "Br": (0.650980392, 0.160784314, 0.160784314),
    "Kr": (0.360784314, 0.721568627, 0.819607843),
    "Rb": (0.439215686, 0.180392157, 0.690196078),
    "Sr": (0, 1, 0),
    "Y": (0.580392157, 1, 1),
    "Zr": (0.580392157, 0.878431373, 0.878431373),
    "Nb": (0.450980392, 0.760784314, 0.788235294),
    "Mo": (0.329411765, 0.709803922, 0.709803922),
    "Tc": (0.231372549, 0.619607843, 0.619607843),
    "Ru": (0.141176471, 0.560784314, 0.560784314),
    "Rh": (0.039215686, 0.490196078, 0.549019608),
    "Pd": (0, 0.411764706, 0.521568627),
    "Ag": (0.752941176, 0.752941176, 0.752941176),
    "Cd": (1, 0.850980392, 0.560784314),
    "In": (0.650980392, 0.458823529, 0.450980392),
    "Sn": (0.4, 0.501960784, 0.501960784),
    "Sb": (0.619607843, 0.388235294, 0.709803922),
    "Te": (0.831372549, 0.478431373, 0),
    "I": (0.580392157, 0, 0.580392157),
    "Xe": (0.258823529, 0.619607843, 0.690196078),
    "Cs": (0.341176471, 0.090196078, 0.560784314),
    "Ba": (0, 0.788235294, 0),
    "La": (0.439215686, 0.831372549, 1),
    "Ce": (1, 1, 0.780392157),
    "Pr": (0.850980392, 1, 0.780392157),
    "Nd": (0.780392157, 1, 0.780392157),
    "Pm": (0.639215686, 1, 0.780392157),
    "Sm": (0.560784314, 1, 0.780392157),
    "Eu": (0.380392157, 1, 0.780392157),
    "Gd": (0.270588235, 1, 0.780392157),
    "Tb": (0.188235294, 1, 0.780392157),
    "Dy": (0.121568627, 1, 0.780392157),
    "Ho": (0, 1, 0.611764706),
    "Er": (0, 0.901960784, 0.458823529),
    "Tm": (0, 0.831372549, 0.321568627),
    "Yb": (0, 0.749019608, 0.219607843),
    "Lu": (0, 0.670588235, 0.141176471),
    "Hf": (0.301960784, 0.760784314, 1),
    "Ta": (0.301960784, 0.650980392, 1),
    "W": (0.129411765, 0.580392157, 0.839215686),
    "Re": (0.149019608, 0.490196078, 0.670588235),
    "Os": (0.149019608, 0.4, 0.588235294),
    "Ir": (0.090196078, 0.329411765, 0.529411765),
    "Pt": (0.815686275, 0.815686275, 0.878431373),
    "Au": (1, 0.819607843, 0.137254902),
    "Hg": (0.721568627, 0.721568627, 0.815686275),
    "Tl": (0.650980392, 0.329411765, 0.301960784),
    "Pb": (0.341176471, 0.349019608, 0.380392157),
    "Bi": (0.619607843, 0.309803922, 0.709803922),
    "Po": (0.670588235, 0.360784314, 0),
    "At": (0.458823529, 0.309803922, 0.270588235),
    "Rn": (0.258823529, 0.509803922, 0.588235294),
    "Fr": (0.258823529, 0, 0.4),
    "Ra": (0, 0.490196078, 0),
    "Ac": (0.439215686, 0.670588235, 0.980392157),
    "Th": (0, 0.729411765, 1),
    "Pa": (0, 0.631372549, 1),
    "U": (0, 0.560784314, 1),
    "Np": (0, 0.501960784, 1),
    "Pu": (0, 0.419607843, 1),
    "Am": (0.329411765, 0.360784314, 0.949019608),
    "Cm": (0.470588235, 0.360784314, 0.890196078),
    "Bk": (0.541176471, 0.309803922, 0.890196078),
    "Cf": (0.631372549, 0.211764706, 0.831372549),
    "Es": (0.701960784, 0.121568627, 0.831372549),
    "Fm": (0.701960784, 0.121568627, 0.729411765),
    "Md": (0.701960784, 0.050980392, 0.650980392),
    "No": (0.741176471, 0.050980392, 0.529411765),
    "Lr": (0.780392157, 0, 0.4),
    "Rf": (0.8, 0, 0.349019608),
    "Db": (0.819607843, 0, 0.309803922),
    "Sg": (0.850980392, 0, 0.270588235),
    "Bh": (0.878431373, 0, 0.219607843),
    "Hs": (0.901960784, 0, 0.180392157),
    "Mt": (0.921568627, 0, 0.149019608),
}

#### in-line execution
def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("path", help="Path to directory", type=str)
    parser.add_argument("title", help="title of plot", type=str)

    # Optional arguments
    parser.add_argument(
        "-i",
        "--interactive",
        help="interactive plotting mode",
        type=bool,
        default=False,
    )
    parser.add_argument("-t", "--title", help="Name of the file", type=str, default="")
    parser.add_argument(
        "-yl",
        "--ylimits",
        nargs="+",
        help="list of upper and lower y-axis limits",
        type=float,
        default=[],
    )
    parser.add_argument("--task", help="Which DOS to plot?", type=str, default="total")
    # Parse arguments
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse the arguments
    args = parseArguments()
    cwd = Path.cwd()
    if args.interactive == True:
        plt.ion()
    else:
        plt.ioff()

    dos = DOS(cwd.joinpath(args.path))
    figure = dos.plot_all_atomic_dos()

    if args.interactive == False:
        os.chdir(str(cwd.joinpath(args.path)))
        plt.savefig(args.title + ".png", dpi=300, bbox_inches="tight")
        os.chdir(str(cwd))
    else:
        plt.show(block=True)

