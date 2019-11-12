# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 17:11:40 2019

@author: Roman Kempt
"""

import numpy as np
import glob, sys, os, math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import argparse
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from pathlib import Path as Path
from scipy import interpolate
import ase.io, ase.cell


plt.style.use("seaborn-ticks")

Angstroem_to_bohr = 1.889725989


class bandstructure:
    """ Band structure object.
    
    Contains all information about a single band structure instance, such as the energy spectrum, the band gap, Fermi level etc.

    Example:    
        >>> from AIMS_tools import bandstructure
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> bs = bandstructure.bandstructure("outputfile")
        >>> bs.plot()
        >>> plt.show()
        >>> plt.savefig("Name.png", dpi=300, transparent=False, bbox_inches="tight", facecolor="white")

    Args:    
        outputfile (str): Path to outputfile.
        get_SOC (bool): Retrieve spectrum with or without spin-orbit coupling (True/False), if calculated.
        custom_path (str): Custom high-symmetry path for plotting, e.g., "G-X-M".
        shift_to (str): Shifts Fermi level. Options are None (default for metallic systems), "middle" for middle of band gap, and "VBM" for valence band maximum.
    
    Attributes:
        path (pathlib object): Directory of outputfile.
        active_SOC (bool): If spin-orbit coupling was included in the control.in file.
        active_GW (bool): If GW was included in the control.in file.
        shift_type (str): Argument of shift_to.
        ksections (dict): Dictionary of path segments and corresponding band file.
        bandfiles (list): List of path objects to band files.
        kpath (list): K-path labels following AFLOW conventions.
        kvectors (dict): Dictionary of k-point labels and fractional coordinates.
        klabel_coords (list): List of x-positions of the k-path labels.
        fermi_level (float): Fermi level energy value in eV.
        band_gap (float): Band gap energy in eV.
        smallest_direct_gap (str): Smallest direct gap from outputfile.
        spectrum (numpy array): Array of k-values and eigenvalues.
        atoms (dict): Dictionary of atom index and label.
        cell (Atoms): ASE atoms object of the geometry.
        rec_cell_lengths (numpy array): Reciprocal lattice vector lengths in 2 pi/bohr.
    
    Note:
        Input coordinates are assumed to be in Angström. Units are converted to atomic units.

    Todo:
        Invoking the class easier by looking for outputfiles automatically.
    """

    def __init__(self, outputfile, get_SOC=True, custom_path="", shift_to="middle"):
        """ Creates band structure instance. """
        cwd = Path.cwd()
        self.path = cwd.joinpath(
            Path(outputfile).parent
        )  # this retrieves the directory where the output file is
        self.shift_type = shift_to
        self.__read_control()
        self.__read_geometry()
        self.__get_output_information(outputfile)
        self.__get_bandfiles(get_SOC)
        self.ksections = dict(zip(self.ksections, self.bandfiles))
        self.kpath = [i[0] for i in self.ksections.keys()]
        self.kpath += [
            list(self.ksections.keys())[-1][1]
        ]  # retrieves the endpoint of the path
        if custom_path != "":
            self.custom_path(custom_path)
        self.__concat_bandfiles(self.bandfiles)
        self.spectrum = self.shift_to(self.spectrum)

    def plot(
        self,
        title="",
        fig=None,
        axes=None,
        color="k",
        var_energy_limits=1.0,
        fix_energy_limits=[],
        mark_gap="lightgray",
        kwargs={},
    ):
        """Plots a band structure instance.
            
            Args:
                title (str): Title of the plot.
                fig (matplotlib figure): Figure to draw the plot on.
                axes (matplotlib axes): Axes to draw the plot on.
                color (str): Color of the lines.
                var_energy_limits (int): Variable energy range above and below the band gap to show.
                fix_energy_limits (list): List of lower and upper energy limits to show.
                mark_gap (str): Color to fill the band gap with or None.
                **kwargs (dict): Passed to matplotlib plotting function.

            Returns:
                axes: matplotlib axes object"""
        if fig == None:
            fig = plt.figure(figsize=(len(self.kpath) / 2, 3))
        if axes == None:
            axes = plt.gca()
        x = self.spectrum[:, 0]
        y = self.spectrum[:, 1:]
        VBM = np.max(y[y < 0])
        CBM = np.min(y[y > 0])
        axes.plot(x, y, color=color, **kwargs)
        if fix_energy_limits == []:
            lower_ylimit = VBM - var_energy_limits
            upper_ylimit = CBM + var_energy_limits
        else:
            lower_ylimit = fix_energy_limits[0]
            upper_ylimit = fix_energy_limits[1]
        if (CBM - VBM) > 0.1 and mark_gap != False:
            axes.fill_between(x, VBM, CBM, color=mark_gap, alpha=0.6)
        axes.set_ylim([lower_ylimit, upper_ylimit])
        axes.set_xlim([0, np.max(x)])
        axes.set_xticks(self.klabel_coords)
        xlabels = []
        for i in range(len(self.kpath)):
            if self.kpath[i] == "G":
                xlabels.append("$\Gamma$")
            else:
                xlabels.append(self.kpath[i])
        axes.set_xticklabels(xlabels)
        axes.set_ylabel("E-E$_\mathrm{F}$ [eV]")
        #axes.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axes.set_xlabel("")
        axes.axhline(y=0, color="k", alpha=0.5, linestyle="--")
        axes.grid(which="major", axis="x", linestyle=":")
        axes.set_title(str(title), loc="center")
        return axes

    def properties(self):
        """ Prints out key properties of the band structure. """
        print("Band gap: {:2f} eV (SOC = {})".format(self.band_gap, self.active_SOC))
        print(self.smallest_direct_gap)
        print("Path: ", self.kpath)

    def __read_bandfile(self, bandfile, kstep):
        """ Reads in band.out file and produces array of eigenvalues for the distance between k-points.
            kstep: shifts the x-axis to glue together all sections in one single axis."""
        try:
            array = [line.split() for line in open(bandfile)]
            array = np.array(array, dtype=float)
        except:
            return "File not found."
        """ Popping out the k-vectors of each line """
        kcoords = (
            array[:, 1:4] * self.rec_cell_lengths
        )  # non-relative positions in 2 pi/bohr
        klength = np.linalg.norm(kcoords[0, :] - kcoords[-1, :])
        klength = np.linspace(0, klength, num=len(kcoords)) + kstep
        new_kmax = np.max(klength)
        """ Removing kcoords and occupations from .out"""
        array = np.delete(array, [0, 1, 2, 3], 1)
        array = np.delete(array, list(range(0, array.shape[1], 2)), axis=1)
        array = np.insert(array, 0, klength, axis=1)
        return array, new_kmax

    def __concat_bandfiles(self, bandfiles):
        """ Concatenates bandfiles and produces array of x-Axis with eigenvalues."""
        kstep = 0
        klabel_coords = [0.0]  # positions of x-ticks
        list_of_arrays = []
        for bandfile in bandfiles:
            array, kstep = self.__read_bandfile(bandfile, kstep)
            klabel_coords.append(kstep)
            list_of_arrays.append(array)
        self.spectrum = np.concatenate(list_of_arrays)
        self.klabel_coords = klabel_coords

    def __read_geometry(self):
        """ Retrieve atom types and number of atoms from geometry. """
        geometry = self.path.joinpath("geometry.in")
        self.cell = ase.io.read(self.path.joinpath("geometry.in"))
        self.rec_cell = (
            self.cell.get_reciprocal_cell() * 2 * math.pi / Angstroem_to_bohr
        )
        self.rec_cell_lengths = ase.cell.Cell.new(
            self.rec_cell
        ).lengths()  # converting to atomic units 2 pi/bohr
        self.atoms = {}
        i = 1  # index to run over atoms
        with open(geometry, "r") as file:
            for line in file.readlines():
                if "atom" in line:
                    self.atoms[i] = line.split()[-1]
                    i += 1

    def __read_control(self):
        """ Retrieve SOC and k-label information. """
        control = self.path.joinpath("control.in")
        bandlines = []
        self.active_SOC = False
        self.active_GW = False
        with open(control, "r") as file:
            for line in file.readlines():
                read = False if line.startswith("#") else True
                if read:
                    if "output band" in line:
                        bandlines.append(line.split())
                    if "include_spin_orbit" in line:
                        self.active_SOC = True
                    if "qpe_calc" in line and "gw" in line:
                        self.active_GW = True
        self.ksections = []
        self.kvectors = {"G": np.array([0.0, 0.0, 0.0])}
        for entry in bandlines:
            self.ksections.append((entry[-2], entry[-1]))
            self.kvectors[entry[-1]] = np.array(
                [entry[5], entry[6], entry[7]], dtype=float
            )
            self.kvectors[entry[-2]] = np.array(
                [entry[2], entry[3], entry[4]], dtype=float
            )

    def custom_path(self, custompath):
        """ This function takes in a custom path of form K1-K2-K3 for plotting.

        Args:
            custompath (str): Hyphen-separated string of path labels, e.g., "G-M-X".

        Note:
            Only the paths that have been calculated in the control.in can be plotted.
         """
        newpath = custompath.split("-")
        check = [(newpath[i], newpath[i + 1]) for i in range(len(newpath) - 1)]
        for pair in check:
            try:
                self.ksections[pair]
            except KeyError:
                print(
                    "The path {}-{} has not been calculated.".format(pair[0], pair[1])
                )
                break
        self.kpath = newpath
        self.bandfiles = [self.ksections[i] for i in check]
        self.__concat_bandfiles(self.bandfiles)

    def __get_bandfiles(self, get_SOC):
        """Sort bandfiles according to k-path, SOC and GW.
        As you can see, the naming of band files in Aims is terribly inconsistent."""
        if (
            self.active_SOC == False
        ):  ### That's the ZORA case if SOC was not calculated.
            bandfiles = [
                self.path.joinpath("band100" + str(i + 1) + ".out")
                for i in range(len(self.ksections))
                if i < 9
            ]
            bandfiles += [
                self.path.joinpath("band10" + str(i + 1) + ".out")
                for i in range(len(self.ksections))
                if i > 8
            ]  # inconsistent naming of files sucks
        elif (
            self.active_SOC == True and get_SOC == False
        ):  ### That's the ZORA case if SOC was calculated.
            bandfiles = [
                self.path.joinpath("band100" + str(i + 1) + ".out.no_soc")
                for i in range(len(self.ksections))
                if i < 9
            ]
            bandfiles += [
                self.path.joinpath("band10" + str(i + 1) + ".out.no_soc")
                for i in range(len(self.ksections))
                if i > 8
            ]
        elif self.active_SOC == True and get_SOC == True:
            bandfiles = [
                self.path.joinpath("band100" + str(i + 1) + ".out")
                for i in range(len(self.ksections))
                if i < 9
            ]
            bandfiles += [
                self.path.joinpath("band10" + str(i + 1) + ".out")
                for i in range(len(self.ksections))
                if i > 8
            ]
        if (
            self.active_SOC == False and self.active_GW == True
        ):  ### That's the ZORA case with GW.
            bandfiles = [
                self.path.joinpath("GW_band100" + str(i + 1) + ".out")
                for i in range(len(self.ksections))
                if i < 9
            ]
            bandfiles += [
                self.path.joinpath("band10" + str(i + 1) + ".out")
                for i in range(len(self.ksections))
                if i > 8
            ]
        self.bandfiles = bandfiles

    def __get_output_information(self, outputfile):
        """ Retrieve information such as Fermi level and band gap from output file."""
        self.smallest_direct_gap = "Direct gap not determined. This usually happens if the fundamental gap is direct."
        with open(outputfile, "r") as file:
            for line in file:
                if "Chemical potential (Fermi level) in eV" in line:
                    self.fermi_level = float(line.split()[-1])
                if "Smallest direct gap :" in line:
                    self.smallest_direct_gap = line

    def shift_to(self, spectrum):
        """ Shifts Fermi level of spectrum according to shift_type attribute.
        
        Returns:
            array: spectrum attribute

        """
        VBM = np.max(spectrum[:, 1:][spectrum[:, 1:] < 0])
        CBM = np.min(spectrum[:, 1:][spectrum[:, 1:] > 0])
        self.band_gap = CBM - VBM
        if self.band_gap < 0.1:
            shift_type = None
        elif self.shift_type == "middle":
            spectrum[:, 1:] -= (VBM + CBM) / 2
        elif self.shift_type == "VBM":
            spectrum[:, 1:] -= VBM
        return spectrum


class fatbandstructure(bandstructure):
    """ Band structure object. Inherits from bandstructure class. 

    Example:    
        >>> from AIMS_tools import bandstructure
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> if __name__ == "main":
        >>>     bs = bandstructure.fatbandstructure("outputfile", parallel=True)
        >>>     bs.plot_all_orbitals()
        >>>     plt.show()
        >>>     plt.savefig("Name.png", dpi=300, transparent=False, bbox_inches="tight", facecolor="white")
    
    Args:
        filter (list): Only processes list of atom labels, e.g., ["W", "S"].
        parallel (bool) : Employs multiprocessing to speed up the data input. Especially useful for systems above 20 atoms.

    
    Attributes:
        filter (list): List of atom labels.
        tot_contributions (dict): Dictionary of numpy arrays per atom containing total contributions.
        s_contributions (dict): Dictionary of numpy arrays per atom containing s-orbital contributions.
        p_contributions (dict): Dictionary of numpy arrays per atom containing p-orbital contributions.
        d_contributions (dict): Dictionary of numpy arrays per atom containing d-orbital contributions.
        mlk_bandfiles (list): List of pathlib objects pointing to the mulliken band output files.

    Note:
        Invoking the parallel option strictly requires your execution script to be safeguarded with the statement: \n
        if __name__ == "__main__": \n
        Hence, this option can not be invoked interactively.

    Todo:
        * Invoking the class easier by looking for outputfiles automatically.
        * Improving fatband plots.
        * Adding a scatter option.
    """

    def __init__(
        self,
        outputfile,
        get_SOC=True,
        custom_path="",
        shift_to="middle",
        filter=[],
        parallel=False,
    ):
        super().__init__(
            outputfile, get_SOC=get_SOC, custom_path=custom_path, shift_to=shift_to
        )
        """ get_SOC is true because for mulliken bands, both spin channels are
        written to the same file. """
        self.filter = list(self.atoms.values()) if filter == [] else filter
        self.atoms = {k: v for k, v in self.atoms.items() if v in self.filter}
        self.tot_contributions = {}
        self.s_contributions = {}
        self.p_contributions = {}
        self.d_contributions = {}
        self.__get_mlk_bandfiles(get_SOC)
        self.ksections = dict(zip(self.ksections, self.mlk_bandfiles))
        if custom_path != "":
            self.custom_path(custom_path, parallel=parallel)
        self.concat_mlk_bandfiles(
            self.mlk_bandfiles, filter=self.filter, parallel=parallel
        )

    def __get_mlk_bandfiles(self, get_SOC):
        """Sort bandfiles that have mulliken information.
        As you can see, the naming of band files in Aims is terribly inconsistent."""
        if (
            self.active_SOC == False
        ):  ### That's the ZORA case if SOC was not calculated.
            bandfiles = [
                self.path.joinpath("bandmlk100" + str(i + 1) + ".out")
                for i in range(len(self.ksections))
                if i < 9
            ]
            bandfiles += [
                self.path.joinpath("bandmlk10" + str(i + 1) + ".out")
                for i in range(len(self.ksections))
                if i > 8
            ]  # inconsistent naming of files sucks
        elif (
            self.active_SOC == True and get_SOC == False
        ):  ### That's the ZORA case if SOC was calculated. For mlk calculations it doesn't exist.
            bandfiles = [
                self.path.joinpath("bandmlk100" + str(i + 1) + ".out.no_soc")
                for i in range(len(self.ksections))
                if i < 9
            ]
            bandfiles += [
                self.path.joinpath("bandmlk10" + str(i + 1) + ".out.no_soc")
                for i in range(len(self.ksections))
                if i > 8
            ]
        elif self.active_SOC == True and get_SOC == True:
            bandfiles = [
                self.path.joinpath("bandmlk100" + str(i + 1) + ".out")
                for i in range(len(self.ksections))
                if i < 9
            ]
            bandfiles += [
                self.path.joinpath("bandmlk10" + str(i + 1) + ".out")
                for i in range(len(self.ksections))
                if i > 8
            ]
        if (
            self.active_SOC == False and self.active_GW == True
        ):  ### That's the ZORA case with GW.
            bandfiles = [
                self.path.joinpath("GW_bandmlk100" + str(i + 1) + ".out")
                for i in range(len(self.ksections))
                if i < 9
            ]
            bandfiles += [
                self.path.joinpath("GW_bandmlk10" + str(i + 1) + ".out")
                for i in range(len(self.ksections))
                if i > 8
            ]
        self.mlk_bandfiles = bandfiles

    def __read_mlk_bandfile(self, bandfile, kstep, atom_number, orbital=4):
        """ Reads in mulliken bandfile and returns eigenvalues and 
        orbital-wise contributions as numpy arrays.
        orbital : int 
        4 --> total, 5 --> s, 6 --> p, 7 --> d ..."""
        with open(bandfile, "r") as file:
            content = file.read()

        split_by_k = [
            z.split("\n") for z in content.split("k point number:") if z != ""
        ]

        energies = []
        tot_contributions = []
        s_contributions = []
        p_contributions = []
        d_contributions = []
        kpoints = []
        for k in split_by_k:
            content = [i.split() for i in k if "State" not in i and i != ""]
            kx, ky, kz = [float(i) for i in content.pop(0)[2:5]]
            kpoints.append([kx, ky, kz])
            empty = np.zeros([len(content), 13])
            for i, j in enumerate(
                content
            ):  # this circumvenes that there are different amounts of columns
                empty[i][0 : len(j)] = j
            content = np.array(empty, dtype=float)
            content = content[content[:, 3] == atom_number]  # filter by atom
            if self.active_SOC == True:
                spin1 = content[content[:, 4] == 1]  # filter by spin one
                spin2 = content[content[:, 4] == 2]  # filter by spin two
                energies.append(np.concatenate([spin1[:, 1], spin2[:, 1]], axis=0))
                tot_contributions.append(
                    np.concatenate([spin1[:, 4 + 1], spin2[:, 4 + 1]], axis=0)
                )
                s_contributions.append(
                    np.concatenate([spin1[:, 5 + 1], spin2[:, 5 + 1]], axis=0)
                )
                p_contributions.append(
                    np.concatenate([spin1[:, 6 + 1], spin2[:, 6 + 1]], axis=0)
                )
                d_contributions.append(
                    np.concatenate([spin1[:, 7 + 1], spin2[:, 7 + 1]], axis=0)
                )
            else:
                energies.append(content[:, 1])
                tot_contributions.append(content[:, 4])  # filter contribution
                s_contributions.append(content[:, 5])  # filter contribution
                p_contributions.append(content[:, 6])  # filter contribution
                d_contributions.append(content[:, 7])  # filter contribution

        kpoints = np.array(kpoints)
        klength = np.linalg.norm(kpoints[0, :] - kpoints[-1, :])
        klength = np.linspace(0, klength, num=len(kpoints)) + kstep
        new_kmax = np.max(klength)

        spectrum = np.insert(np.array(energies), 0, klength, axis=1)
        return (
            new_kmax,
            spectrum,
            tot_contributions,
            s_contributions,
            p_contributions,
            d_contributions,
        )

    def concat_atom(self, atom, mlk_bandfiles, return_dict):
        """ This is a helper function for parallelisation."""
        kstep = 0
        energies = []
        tot_contributions = []
        s_contributions = []
        p_contributions = []
        d_contributions = []
        klabel_coords = [kstep]
        for bandfile in mlk_bandfiles:
            kstep, energy, tot_contribution, s_contribution, p_contribution, d_contribution = self.__read_mlk_bandfile(
                bandfile, kstep, atom
            )
            klabel_coords.append(kstep)
            energies.append(energy)
            tot_contributions.append(tot_contribution)
            s_contributions.append(s_contribution)
            p_contributions.append(p_contribution)
            d_contributions.append(d_contribution)
            print(
                "Processed {} for atom {} Nr. {} .".format(
                    bandfile, self.atoms[atom], atom
                )
            )
        tot, s, p, d = {}, {}, {}, {}
        tot[atom] = np.concatenate(tot_contributions, axis=0)
        s[atom] = np.concatenate(s_contributions, axis=0)
        p[atom] = np.concatenate(p_contributions, axis=0)
        d[atom] = np.concatenate(d_contributions, axis=0)
        if atom == list(self.atoms.keys())[0]:
            spectrum = np.concatenate(energies, axis=0)
            spectrum = self.shift_to(spectrum)
            return_dict[atom] = [
                klabel_coords,
                spectrum,
                tot[atom],
                s[atom],
                p[atom],
                d[atom],
            ]
        else:
            return_dict[atom] = [None, None, tot[atom], s[atom], p[atom], d[atom]]

    def concat_mlk_bandfiles(self, mlk_bandfiles, filter, parallel=False):
        """Concatenates contributions and energies for every single k-point, atom, and band."""
        import multiprocessing
        from multiprocessing import Process

        if parallel == False:
            print("Reading in mulliken band files in serial...")
            return_dict = dict()
            for atom in self.atoms.keys():
                self.concat_atom(atom, mlk_bandfiles, return_dict)
        if parallel == True:
            print("Reading in mulliken band files in parallel.")
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            processes = []
            for atom in self.atoms.keys():
                p = Process(
                    target=self.concat_atom, args=[atom, mlk_bandfiles, return_dict]
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        self.klabel_coords = return_dict[list(self.atoms.keys())[0]][0]
        self.spectrum = return_dict[list(self.atoms.keys())[0]][1]
        for atom in self.atoms.keys():
            self.tot_contributions[atom] = return_dict[atom][2]
            self.s_contributions[atom] = return_dict[atom][3]
            self.p_contributions[atom] = return_dict[atom][4]
            self.d_contributions[atom] = return_dict[atom][5]

    def custom_path(self, custompath, parallel=False):
        """ This function takes in a custom path of form K1-K2-K3 for plotting.

        Args:
            custompath (str): Hyphen-separated string of path labels, e.g., "G-M-X".

        Note:
            Only the paths that have been calculated in the control.in can be plotted.
         """
        newpath = custompath.split("-")
        check = [(newpath[i], newpath[i + 1]) for i in range(len(newpath) - 1)]
        for pair in check:
            try:
                self.ksections[pair]
            except KeyError:
                print(
                    "The path {}-{} has not been calculated.".format(pair[0], pair[1])
                )
                break
        self.kpath = newpath
        self.mlk_bandfiles = [self.ksections[i] for i in check]
        self.concat_mlk_bandfiles(self.mlk_bandfiles, parallel=parallel)

    def plot(
        self,
        contribution,
        title="",
        fig=None,
        axes=None,
        cmap="Blues",
        var_energy_limits=1.0,
        fix_energy_limits=[],
        nbands=False,
        interpolation_step=False,
        kwargs={},
    ):
        """ Plots a fat band structure instance.
        
        Args:
            contribution (numpy array): Atomic contribution attribute, e.g., tot_contribution["C"].
            title (str): Title of the plot.
            fig (matplotlib figure): Figure to draw the plot on.
            axes (matplotlib axes): Axes to draw the plot on.
            cmap (str): Matplotlib colormap instance.
            var_energy_limits (int): Variable energy range above and below the band gap to show.
            fix_energy_limits (list): List of lower and upper energy limit to show.
            nbands (int): False or integer. Number of bands above and below the Fermi level to colorize.
            interpolation_step (float): False or float. Performs 1D interpolation of every band with the specified
            step size (e.g., 0.001). May cause substantial computational effort.
            **kwargs (dict): Currently not used.
        
        Returns:
            axes: matplotlib axes object"""
        if fig == None:
            fig = plt.figure(figsize=(len(self.kpath) / 2, 3))
        if axes == None:
            axes = plt.gca()
        x = self.spectrum[:, 0]
        y = self.spectrum[:, 1:]
        VBM = np.max(y[y < 0])
        CBM = np.min(y[y > 0])

        if fix_energy_limits == []:
            lower_ylimit = VBM - var_energy_limits
            upper_ylimit = CBM + var_energy_limits
        else:
            lower_ylimit = fix_energy_limits[0]
            upper_ylimit = fix_energy_limits[1]

        if nbands != False:
            index = np.where(y == VBM)
            if self.active_SOC == True:
                col = index[1][[0, -1]]
                r1 = [i for i in range(col[0] - nbands, col[0] + nbands + 1)]
                r2 = [i for i in range(col[1] - nbands, col[1] + nbands + 1)]
                y = y[:, r1 + r2]
                contribution = contribution[
                    :, [i - 1 for i in r1] + [i - 1 for i in r1]
                ]  # contributions have no k-values, hence one column less
            else:
                col = index[1][0]
                y = y[:, col - nbands : col + nbands + 1]
                contribution = contribution[:, col - nbands - 1 : col + nbands]

        for band in range(y.shape[1]):
            band_x = x
            band_y = y[:, band]
            band_width = contribution[:, band - 1]
            if interpolation_step != False:
                f1 = interpolate.interp1d(x, band_y)
                f2 = interpolate.interp1d(x, band_width)
                band_x = np.arange(0, np.max(x), interpolation_step)
                band_y = f1(band_x)
                band_width = f2(band_x)
            band_width = band_width[:-1]
            points = np.array([band_x, band_y]).T.reshape(-1, 1, 2)
            segments = np.concatenate(
                [points[:-1], points[1:]], axis=1
            )  # this reshapes it into (x1, x2) (y1, y2) pairs
            ### giving them color
            cmap = plt.get_cmap(cmap)
            my_cmap = cmap(np.arange(cmap.N))
            my_cmap[:, -1] = np.linspace(0, 1, cmap.N)  # this adds alpha
            my_cmap = ListedColormap(my_cmap)
            lc = LineCollection(
                segments,
                linewidths=band_width * 2.5,  # making them fat
                cmap=my_cmap,
                norm=plt.Normalize(0, 1),
                capstyle="butt",
            )
            lc.set_array(band_width)
            axes.add_collection(lc)

        axes.set_ylim([lower_ylimit, upper_ylimit])
        axes.set_xlim([0, np.max(x)])
        axes.set_xticks(self.klabel_coords)
        xlabels = []
        for i in range(len(self.kpath)):
            if self.kpath[i] == "G":
                xlabels.append("$\Gamma$")
            else:
                xlabels.append(self.kpath[i])
        axes.set_xticklabels(xlabels)
        axes.set_ylabel("E-E$_\mathrm{F}$ [eV]")
        axes.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axes.set_xlabel("")
        axes.axhline(y=0, color="k", alpha=0.5, linestyle="--")
        axes.grid(which="major", axis="x", linestyle=":")
        axes.set_title(str(title), loc="center")
        return axes

    def sum_contributions(self):
        """ Sums (normalized) atomic contributions for the same species.

        This method is optional, since the atoms are not always symmetry equivalent.
        Atoms attribute will be modified and ordered by heaviest atoms.
        """
        atoms = self.atoms
        reverse_atoms = {}
        for key, value in atoms.items():
            reverse_atoms.setdefault(value, set()).add(key)
        duplicates = [
            values for key, values in reverse_atoms.items() if len(values) > 1
        ]
        for duplicate in duplicates:
            number = len(duplicate)
            new_key = list(duplicate)[0]
            new_contribution = np.zeros(self.tot_contributions[new_key].shape)
            for key in duplicate:
                new_contribution += self.tot_contributions.pop(key)
                if key != new_key:
                    self.atoms.pop(key)
            new_contribution = new_contribution
            self.tot_contributions[new_key] = new_contribution
        ## reordering atoms dictionary
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
        keys = list(self.atoms.keys())
        vals = list(self.atoms.values())
        pse_vals = [pse[val] for val in vals]
        sorted_keys = [
            x for _, x in sorted(zip(pse_vals, keys), key=lambda pair: pair[0])
        ]
        self.atoms = {key: self.atoms[key] for key in sorted_keys}

    def plot_all_species(
        self,
        axes=None,
        fig=None,
        title="",
        sum=True,
        var_energy_limits=1.0,
        fix_energy_limits=[],
        nbands=False,
        interpolation_step=False,
        kwargs={},
    ):
        """ Plots a fatbandstructure instance with all species overlaid.
        
        Shares attributes with the in-built plot function. Gives an error if invoked with more than 5 species.

        .. image:: ../pictures/MoSe2_fatband_species.png
            :width: 220px
            :align: center
            :height: 250px
        
        Args:
            sum (bool): True or False. Invokes sum_contribution() method.
            **kwargs (dict): Keyword arguments are passed to the plot() method.
        
        Returns:
            axes: matplotlib axes object"""
        if fig == None:
            fig = plt.figure(figsize=(len(self.kpath) / 2, 4))
        if axes != None:
            axes = plt.gca()
        else:
            axes = plt.subplot2grid((1, 1), (0, 0), fig=fig)
        if sum == True:
            self.sum_contributions()
        if len(self.atoms.keys()) > 5:
            print(
                """Humans can't perceive enough colors to make a band structure plot
            possible with {} species.""".format(
                    len(self.atoms.keys())
                )
            )
            sys.exit()
        if interpolation_step != False:
            print(
                """Performing 1D interpolation of every single band to obtain a smoother plot.\nThis computation could take a while and is only recommended for production quality pictures."""
            )
        cmaps = ["Blues", "Oranges", "Greens", "Purples", "Reds"]
        colors = ["darkblue", "darkorange", "darkgreen", "darkviolet", "darkred"]
        handles = []
        i = 0
        for atom in self.atoms.keys():
            label = self.atoms[atom]
            self.plot(
                self.tot_contributions[atom],
                cmap=cmaps[i],
                axes=axes,
                fig=fig,
                title=title,
                var_energy_limits=var_energy_limits,
                fix_energy_limits=fix_energy_limits,
                nbands=nbands,
                interpolation_step=interpolation_step,
                **kwargs
            )
            handles.append(Line2D([0], [0], color=colors[i], label=label, lw=1.5))
            i += 1
        lgd = axes.legend(
            handles=handles,
            frameon=True,
            fancybox=False,
            borderpad=0.4,
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )
        return axes

    def plot_all_orbitals(
        self,
        axes=None,
        fig=None,
        title="",
        var_energy_limits=1.0,
        fix_energy_limits=[],
        nbands=False,
        interpolation_step=False,
        kwargs={},
    ):
        """ Plots a fatbandstructure instance with all orbital characters overlaid.
        
        Shares attributes with the in-built plot function.

        .. image:: ../pictures/MoSe2_fatband_orbitals.png
            :width: 220px
            :align: center
            :height: 250px

        Args:
            **kwargs (dict): Keyword arguments are passed to the plot() method.
                
        Returns:
            axes: matplotlib axes object """
        if fig == None:
            fig = plt.figure(figsize=(len(self.kpath) / 2, 4))
        if axes != None:
            axes = plt.gca()
        else:
            axes = plt.subplot2grid((1, 1), (0, 0), fig=fig)
        if interpolation_step != False:
            print(
                """Performing 1D interpolation of every single band to obtain a smoother plot.\nThis computation could take a while and is only recommended for production quality pictures."""
            )

        cmaps = ["Oranges", "Blues", "Greens", "Purples", "Reds"]
        colors = ["darkorange", "darkblue", "darkgreen", "darkviolet", "darkred"]
        handles = []
        i = 0
        orbitals = {}
        orbitals["s"] = np.sum(
            [
                self.s_contributions[atom]
                for atom in self.atoms.keys()
                if self.atoms[atom] in self.filter
            ],
            axis=0,
        )
        orbitals["p"] = np.sum(
            [
                self.p_contributions[atom]
                for atom in self.atoms.keys()
                if self.atoms[atom] in self.filter
            ],
            axis=0,
        )
        orbitals["d"] = np.sum(
            [
                self.d_contributions[atom]
                for atom in self.atoms.keys()
                if self.atoms[atom] in self.filter
            ],
            axis=0,
        )
        for orbital in orbitals.keys():
            self.plot(
                orbitals[orbital],
                cmap=cmaps[i],
                axes=axes,
                fig=fig,
                title=title,
                var_energy_limits=var_energy_limits,
                fix_energy_limits=fix_energy_limits,
                nbands=nbands,
                interpolation_step=interpolation_step,
                **kwargs
            )
            handles.append(Line2D([0], [0], color=colors[i], label=orbital, lw=1.5))
            i += 1
        lgd = axes.legend(
            handles=handles,
            frameon=True,
            fancybox=False,
            borderpad=0.4,
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )
        return axes


# def parseArguments():
#     # Create argument parser
#     parser = argparse.ArgumentParser()

#     # Positional mandatory arguments
#     parser.add_argument("path", help="Path to directory", type=str)
#     parser.add_argument("title", help="Name of the file", type=str)

#     # Optional arguments
#     parser.add_argument(
#         "--ZORA_color", help="ZORA bands color", type=str, default="black"
#     )
#     parser.add_argument(
#         "--SOC_color", help="SOC bands color", type=str, default="crimson"
#     )
#     parser.add_argument(
#         "-i",
#         "--interactive",
#         help="interactive plotting mode",
#         type=bool,
#         default=False,
#     )
#     parser.add_argument(
#         "-yl",
#         "--ylimits",
#         nargs="+",
#         help="list of upper and lower y-axis limits",
#         type=float,
#         default=[],
#     )
#     # Parse arguments
#     args = parser.parse_args()
#     return args


# if __name__ == "__main__":
#     # Parse the arguments
#     args = parseArguments()
#     if args.interactive == True:
#         plt.ion()
#     else:
#         plt.ioff()

#     ZORA = bandstructure(args.path, get_SOC=False)
#     if ZORA.active_SOC == True:
#         plot_ZORA_and_SOC(
#             args.path,
#             fix_energy_limits=args.ylimits,
#             ZORA_color=args.ZORA_color,
#             SOC_color=args.SOC_color,
#         )
#     else:
#         ZORA.plot(
#             title=args.title, fix_energy_limits=args.ylimits, color=args.ZORA_color
#         )
#         ZORA.properties()
#     if args.interactive == False:
#         plt.savefig(os.path.basename(args.title) + ".png", dpi=300, bbox_inches="tight")
#     else:
#         plt.show(block=True)
