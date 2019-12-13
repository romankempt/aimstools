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
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path as Path
from scipy import interpolate
import ase.io, ase.cell
from AIMS_tools.postprocessing import postprocess

plt.style.use("seaborn-ticks")
plt.rcParams["legend.handlelength"] = 0.8
plt.rcParams["legend.framealpha"] = 0.8
font_name = "Arial"
font_size = 8.5
plt.rcParams.update({"font.sans-serif": font_name, "font.size": font_size})

class bandstructure(postprocess):
    """ Band structure object. Inherits from postprocess.
    
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
        spin (int): Retrieve spin channel 1 or 2. Defaults to None (spin-restricted) or 1 (collinear).
        shift_to (str): Shifts Fermi level. Options are None (default for metallic systems), "middle" for middle of band gap, and "VBM" for valence band maximum.
    
    Attributes:
        shift_type (str): Argument of shift_to.
        ksections (dict): Dictionary of path segments and corresponding band file.
        bandsegments (dict): Dictionary of path segments and corresponding eigenvalues.
        kpath (list): K-path labels following AFLOW conventions.
        kvectors (dict): Dictionary of k-point labels and fractional coordinates.
        klabel_coords (list): List of x-positions of the k-path labels.
        band_gap (float): Band gap energy in eV.
        spectrum (numpy array): Array of k-values and eigenvalues.
  
    """

    def __init__(self, outputfile, get_SOC=True, spin=None, shift_to="middle"):
        super().__init__(outputfile, get_SOC=get_SOC, spin=spin)
        self.shift_type = shift_to
        self.__get_bandfiles(get_SOC)
        self.ksections = dict(zip(self.bandfiles, self.ksections))
        self.kpath = [i[0] for i in self.ksections.values()]
        self.kpath += [
            list(self.ksections.values())[-1][1]
        ]  # retrieves the endpoint of the path
        self.bandsegments = self.__read_bandfiles()
        self.__create_spectrum()

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
        if (CBM - VBM) > 0.1 and (mark_gap != False) and (self.spin == None):
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
        ylocs = ticker.MultipleLocator(
            base=0.5
        )  # this locator puts ticks at regular intervals
        axes.yaxis.set_major_locator(ylocs)
        axes.set_xlabel("")
        axes.axhline(y=0, color="k", alpha=0.5, linestyle="--")
        axes.grid(which="major", axis="x", linestyle=":")
        axes.set_title(str(title), loc="center")
        return axes

    def properties(self):
        """ Prints out key properties of the band structure. """
        print("Sum Formula: {}".format(self.atoms.symbols))
        print("Number of k-points for SCF: {}".format(self.k_points))
        cell = self.atoms.cell
        if np.array_equal(cell[2], [0.0, 0.0, 100.0]):
            pbc = 2
            area = np.linalg.norm(np.cross(cell[0], cell[1]))
            kdens = self.k_points / area
        else:
            pbc = 3
            volume = self.atoms.get_volume()
            kdens = self.k_points / volume

        if pbc == 2:
            print(
                "System seems to be 2D. The k-point density is {:.4f} points/bohr^2 .".format(
                    kdens
                )
            )
        else:
            print(
                "System seems to be 3D. The k-point density is {:.4f} points/bohr^3 .".format(
                    kdens
                )
            )

        print(
            "Band gap: {:2f} eV (SOC = {}, spin channel = {})".format(
                self.band_gap, self.active_SOC, self.spin
            )
        )
        print(self.smallest_direct_gap)
        print("Path: ", self.kpath)
        import ase.spacegroup

        brav_latt = self.atoms.cell.get_bravais_lattice(pbc=self.atoms.pbc)
        sg = ase.spacegroup.get_spacegroup(self.atoms)
        print("Space group: {} (Nr. {})".format(sg.symbol, sg.no))
        print("Bravais lattice: {}".format(brav_latt))

    def __read_bandfiles(self):
        """ Reads in band.out files."""
        bandsegments = {}
        for bandfile in self.ksections.keys():
            try:
                array = [line.split() for line in open(bandfile)]
                array = np.array(array, dtype=float)
            except:
                return "File {} not found.".format(bandfile)
            array[:, 1:4] * self.rec_cell_lengths  # non-relative positions in bohr^(-1)
            # Removing index and occupations from .out
            array = np.delete(array, [0] + list(range(4, array.shape[1], 2)), axis=1)
            bandsegments[self.ksections[bandfile]] = array
        return bandsegments

    def __create_spectrum(self):
        """ Merges bandsegments to a single spectrum with suitable x-axis."""
        kstep = 0
        klabel_coords = [0.0]  # positions of x-ticks
        specs = []
        segments = [
            (self.kpath[i], self.kpath[i + 1]) for i in range(len(self.kpath) - 1)
        ]
        for segment in segments:
            array = self.bandsegments[segment]
            energies = array[:, 3:]
            start = kstep
            kstep += np.linalg.norm(array[-1, [0, 1, 2]] - array[0, [0, 1, 2]])
            kaxis = np.linspace(start, kstep, array.shape[0])
            klabel_coords.append(kstep)
            energies = np.insert(energies, 0, kaxis, axis=1)
            specs.append(energies)
        spectrum = np.concatenate(specs, axis=0)
        self.spectrum = self.shift_to(spectrum)
        self.klabel_coords = klabel_coords

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
                d = dict(
                    zip(list(self.ksections.values()), list(self.ksections.keys()))
                )
                d[pair]
            except KeyError:
                sys.exit(
                    "The path {}-{} has not been calculated.".format(pair[0], pair[1])
                )
        self.kpath = newpath
        self.__create_spectrum()

    def __get_bandfiles(self, get_SOC):
        """Sort bandfiles according to k-path, spin, SOC and GW.
        As you can see, the naming of band files in Aims is terribly inconsistent."""
        if self.spin != None:
            stem = "band" + str(self.spin)
        else:
            stem = "band1"
        if (
            self.active_SOC == False
        ):  ### That's the ZORA case if SOC was not calculated.
            bandfiles = [
                self.path.joinpath(stem + "00" + str(i + 1) + ".out")
                for i in range(len(self.ksections))
                if i < 9
            ]
            bandfiles += [
                self.path.joinpath(stem + "0" + str(i + 1) + ".out")
                for i in range(len(self.ksections))
                if i > 8
            ]  # inconsistent naming of files sucks
        elif (
            self.active_SOC == True and get_SOC == False
        ):  ### That's the ZORA case if SOC was calculated.
            bandfiles = [
                self.path.joinpath(stem + "00" + str(i + 1) + ".out.no_soc")
                for i in range(len(self.ksections))
                if i < 9
            ]
            bandfiles += [
                self.path.joinpath(stem + "0" + str(i + 1) + ".out.no_soc")
                for i in range(len(self.ksections))
                if i > 8
            ]
        elif self.active_SOC == True and get_SOC == True:
            bandfiles = [
                self.path.joinpath(stem + "00" + str(i + 1) + ".out")
                for i in range(len(self.ksections))
                if i < 9
            ]
            bandfiles += [
                self.path.joinpath(stem + "0" + str(i + 1) + ".out")
                for i in range(len(self.ksections))
                if i > 8
            ]
        if (
            self.active_SOC == False and self.active_GW == True
        ):  ### That's the ZORA case with GW.
            bandfiles = [
                self.path.joinpath("GW_" + stem + "00" + str(i + 1) + ".out")
                for i in range(len(self.ksections))
                if i < 9
            ]
            bandfiles += [
                self.path.joinpath("GW_" + stem + "0" + str(i + 1) + ".out")
                for i in range(len(self.ksections))
                if i > 8
            ]
        self.bandfiles = bandfiles

    def shift_to(self, spectrum):
        """ Shifts Fermi level of spectrum according to shift_type attribute.
        
        Returns:
            array: spectrum attribute

        """
        VBM = np.max(spectrum[:, 1:][spectrum[:, 1:] < 0])
        CBM = np.min(spectrum[:, 1:][spectrum[:, 1:] > 0])
        self.band_gap = CBM - VBM
        if (self.band_gap < 0.1) or (self.spin != None):
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
        >>> bs = bandstructure.fatbandstructure("outputfile")
        >>> bs.plot_all_orbitals()
        >>> plt.show()
        >>> plt.savefig("Name.png", dpi=300, transparent=False, bbox_inches="tight", facecolor="white")
    
    Args:
        filter (list): Only processes list of atom labels, e.g., ["W", "S"].
        parallel (bool) : Currently disabled. New serial implementation is much faster than multiprocessing for up to 500 atoms.

    
    Attributes:
        filter (list): List of atom labels.
        tot_contributions (dict): Dictionary of numpy arrays per atom containing total contributions.
        s_contributions (dict): Dictionary of numpy arrays per atom containing s-orbital contributions.
        p_contributions (dict): Dictionary of numpy arrays per atom containing p-orbital contributions.
        d_contributions (dict): Dictionary of numpy arrays per atom containing d-orbital contributions.
        mlk_bandsegments (dict): Nested dictionary of path segments and data.

    Note:
        Invoking the parallel option strictly requires your execution script to be safeguarded with the statement: \n
        if __name__ == "__main__": \n
        Hence, this option can not be invoked interactively.\n
        In the current version, the parallel option is disabled. The new serial implementation \n
        should be much faster for any reasonable system and runs more stably.

    Todo:        
        * Improving fatband plots.
        * Adding a scatter option.
    """

    def __init__(
        self, outputfile, get_SOC=True, spin=None, shift_to="middle", filter=[]
    ):
        super().__init__(outputfile, get_SOC=get_SOC, shift_to=shift_to, spin=spin)
        # get_SOC is true because for mulliken bands, both spin channels are
        # written to the same file.
        self.filter = list(self.atoms.values()) if filter == [] else filter
        self.atoms = {k: v for k, v in self.atoms.items() if v in self.filter}
        self.tot_contributions = {}
        self.s_contributions = {}
        self.p_contributions = {}
        self.d_contributions = {}
        self.__get_mlk_bandfiles(get_SOC)
        self.ksections = dict(zip(self.mlk_bandfiles, list(self.ksections.values())))
        self.__read_mlk_bandfiles()
        self.__create_mlk_spectra()

    def __get_mlk_bandfiles(self, get_SOC):
        # """Sort bandfiles that have mulliken information.
        # As you can see, the naming of band files in Aims is terribly inconsistent."""
        if self.spin != None:
            stem = "bandmlk" + str(self.spin)
        else:
            stem = "bandmlk1"
        if (
            self.active_SOC == False
        ):  ### That's the ZORA case if SOC was not calculated.
            bandfiles = [
                self.path.joinpath(stem + "00" + str(i + 1) + ".out")
                for i in range(len(self.ksections))
                if i < 9
            ]
            bandfiles += [
                self.path.joinpath(stem + "0" + str(i + 1) + ".out")
                for i in range(len(self.ksections))
                if i > 8
            ]  # inconsistent naming of files sucks
        elif (
            self.active_SOC == True and get_SOC == False
        ):  ### That's the ZORA case if SOC was calculated. For mlk calculations it doesn't exist.
            bandfiles = [
                self.path.joinpath(stem + "00" + str(i + 1) + ".out.no_soc")
                for i in range(len(self.ksections))
                if i < 9
            ]
            bandfiles += [
                self.path.joinpath(stem + "0" + str(i + 1) + ".out.no_soc")
                for i in range(len(self.ksections))
                if i > 8
            ]
        elif self.active_SOC == True and get_SOC == True:
            bandfiles = [
                self.path.joinpath(stem + "00" + str(i + 1) + ".out")
                for i in range(len(self.ksections))
                if i < 9
            ]
            bandfiles += [
                self.path.joinpath(stem + "0" + str(i + 1) + ".out")
                for i in range(len(self.ksections))
                if i > 8
            ]
        if (
            self.active_SOC == False and self.active_GW == True
        ):  ### That's the ZORA case with GW.
            bandfiles = [
                self.path.joinpath("GW_" + stem + "00" + str(i + 1) + ".out")
                for i in range(len(self.ksections))
                if i < 9
            ]
            bandfiles += [
                self.path.joinpath("GW_" + stem + "0" + str(i + 1) + ".out")
                for i in range(len(self.ksections))
                if i > 8
            ]
        self.mlk_bandfiles = bandfiles

    def __read_mlk_bandfile(self, bandfile, bandsegments, key):
        # Reads in mulliken bandfile and puts everything in dictionaries.
        try:
            with open(bandfile, "r") as file:
                content = file.read()
        except:
            return "File {} not found.".format(bandfile)
        split_by_k = [
            z.split("\n") for z in content.split("k point number:") if z != ""
        ]
        kpoints = []
        bandsegments[key] = {}
        segment = bandsegments[key]
        for k in split_by_k:  # iterate over k-points
            content = [i.split() for i in k if "State" not in i and i != ""]
            kx, ky, kz = [float(i) for i in content.pop(0)[2:5]]
            kpoints.append([kx, ky, kz])
            array = np.zeros([len(content), 12])
            for i, j in enumerate(
                content
            ):  # this circumvenes that there are different amounts of columns
                array[i][0 : len(j)] = j
            content = np.array(array, dtype=float)
            for atom in self.atoms.keys():
                cons = self.__read_atom_cons(content, atom)
                if atom not in segment.keys():
                    segment[atom] = cons
                else:
                    for entry in segment[atom].keys():
                        segment[atom][entry] = np.vstack(
                            [segment[atom][entry], cons[entry]]
                        )
        kpoints = np.array(kpoints)
        segment["k"] = kpoints
        bandsegments[key] = segment

    def __read_atom_cons(self, content, atom):
        # Reads atomic contributions from k-point
        content = content[content[:, 3] == atom]  # filter by atom
        if self.active_SOC == True:
            spin1 = content[content[:, 4] == 1]  # filter by spin one
            spin2 = content[content[:, 4] == 2]  # filter by spin two
            energies = np.concatenate([spin1[:, 1], spin2[:, 1]], axis=0)
            tot_cons = np.concatenate([spin1[:, 4 + 1], spin2[:, 4 + 1]], axis=0)
            s_cons = np.concatenate([spin1[:, 5 + 1], spin2[:, 5 + 1]], axis=0)
            p_cons = np.concatenate([spin1[:, 6 + 1], spin2[:, 6 + 1]], axis=0)
            d_cons = np.concatenate([spin1[:, 7 + 1], spin2[:, 7 + 1]], axis=0)
        else:
            energies = content[:, 1]
            tot_cons = content[:, 4]  # filter contribution
            s_cons = content[:, 5]  # filter contribution
            p_cons = content[:, 6]  # filter contribution
            d_cons = content[:, 7]  # filter contribution
        return {"ev": energies, "tot": tot_cons, "s": s_cons, "p": p_cons, "d": d_cons}

    def __read_mlk_bandfiles(self, parallel=False):
        # Iterates __read_mlk_bandfile() over bandfiles
        import multiprocessing
        from multiprocessing import Process

        if parallel == False:
            print("Reading in mulliken band files in serial...")
            bandsegments = dict()
            for bandfile in self.ksections.keys():
                print("     Processing {} ...".format(bandfile.name))
                key = self.ksections[bandfile]
                self.__read_mlk_bandfile(bandfile, bandsegments, key)
            self.mlk_bandsegments = bandsegments
        if parallel == True:
            print("    Reading in mulliken band files in parallel...")
            manager = multiprocessing.Manager()
            bandsegments = manager.dict()
            processes = []
            for path in self.ksections.keys():
                bandfile = self.ksections[path]
                print("     Processing {} ...".format(bandfile.name))
                p = Process(
                    target=self.__read_mlk_bandfile, args=[bandfile, bandsegments, path]
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            self.mlk_bandsegments = bandsegments

    def __create_mlk_spectrum(self, atom):
        # creates spectrum for energy and contributions with suitable x-axis
        kstep = 0
        klabel_coords = [0.0]  # positions of x-ticks
        specs = []
        tots = []
        ss = []
        ps = []
        ds = []
        segments = [
            (self.kpath[i], self.kpath[i + 1]) for i in range(len(self.kpath) - 1)
        ]
        for segment in segments:
            data = self.mlk_bandsegments[segment]
            energies = data[atom]["ev"]
            tots.append(data[atom]["tot"])
            ss.append(data[atom]["s"])
            ps.append(data[atom]["p"])
            ds.append(data[atom]["d"])
            start = kstep
            kstep += np.linalg.norm(data["k"][-1, [0, 1, 2]] - data["k"][0, [0, 1, 2]])
            kaxis = np.linspace(start, kstep, data["k"].shape[0])
            klabel_coords.append(kstep)
            energies = np.insert(energies, 0, kaxis, axis=1)
            specs.append(energies)
        spectrum = np.concatenate(specs, axis=0)
        spectrum = self.shift_to(spectrum)
        tots = np.concatenate(tots, axis=0)
        ss = np.concatenate(ss, axis=0)
        ps = np.concatenate(ps, axis=0)
        ds = np.concatenate(ds, axis=0)
        self.klabel_coords = klabel_coords
        return [spectrum, tots, ss, ps, ds]

    def __create_mlk_spectra(self):
        # iterates __create_mlk_spectrum over all bandsegments
        for atom in self.atoms.keys():
            data = self.__create_mlk_spectrum(atom)
            self.spectrum = data[0]
            self.tot_contributions[atom] = data[1]
            self.s_contributions[atom] = data[2]
            self.p_contributions[atom] = data[3]
            self.d_contributions[atom] = data[4]

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
                d = dict(
                    zip(list(self.ksections.values()), list(self.ksections.keys()))
                )
                d[pair]
            except KeyError:
                sys.exit(
                    "The path {}-{} has not been calculated.".format(pair[0], pair[1])
                )
        self.kpath = newpath
        self.create_spectrum()
        self.__create_mlk_spectra()

    def plot_mlk(
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
        ylocs = ticker.MultipleLocator(
            base=0.5
        )  # this locator puts ticks at regular intervals
        axes.yaxis.set_major_locator(ylocs)
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
            self.plot_mlk(
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
            self.plot_mlk(
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

