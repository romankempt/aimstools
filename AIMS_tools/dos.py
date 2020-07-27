import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
import matplotlib.ticker as ticker

import re

from AIMS_tools.misc import *
from AIMS_tools.postprocessing import postprocess


class density_of_states(postprocess):
    """ Density-of-states object.
    
    Contains all information about a DOS instance, such as the DOS per atom. 
    
    Example:    
        >>> from AIMS_tools import dos
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> ds = dos.density_of_states("outputfile")
        >>> ds.plot_all_species()
        >>> plt.show()
        >>> plt.savefig("Name.png", dpi=300, transparent=False, bbox_inches="tight", facecolor="white")

    Args:
        outputfile (str): Path to outputfile.
        get_SOC (bool): Retrieve DOS with or without spin-orbit coupling (True/False), if calculated.
        spin (int): Retrieve spin channel 1 or 2. Defaults to None (spin-restricted) or 1 (collinear).        

    Attributes:
        dos_per_atom (dict): Dictionary of atom labels and density of states as numpy array of energy vs. DOS.
        total_dos (numpy array): Array of energy vs. DOS.
    
    Todo:
        - provide a function to plot orbital-resolved DOS of atom.
    """

    def __init__(self, outputfile, get_SOC=True, spin=None):
        super().__init__(outputfile, get_SOC=get_SOC, spin=spin)
        if self.success == False:
            sys.exit("Calculation did not converge.")
        self.band_gap = self.CBM - self.VBM
        dosfiles = self.__get_dos_files(get_SOC)
        dos_per_atom = []
        for atom in self.structure.species.keys():
            atomic_dos = self.__sort_dosfiles(dosfiles, atom)
            atomic_dos = self.__sum_dosfiles(atomic_dos)
            dos_per_atom.append(atomic_dos)
        self.dos_per_atom = dict(zip(self.structure.species.keys(), dos_per_atom))
        self.total_dos = self.__get_total_dos(self.dos_per_atom)

    def __str__(self):
        return "DOS"

    def __get_dos_files(self, get_SOC):
        """ Get atom_projected dos files.
            get_SOC : True or False to obtain the ones with or without SOC.
            """
        import re

        regex = re.compile(r"^atom_proj(ected)?_dos_[A-Z]([a-z])?\d{4}\.dat")
        dosfiles = [
            str(j) for j in list(self.path.glob("*.dat*")) if regex.match(str(j))
        ]
        if (self.active_SOC == True) and (self.spin == None):
            if get_SOC == True:
                dosfiles = [str(i) for i in dosfiles if "no_soc" not in str(i)]
            elif get_SOC == False:
                dosfiles = [str(i) for i in dosfiles if "no_soc" in str(i)]

        if (self.active_SOC == True) and (self.spin == 1):
            if get_SOC == True:
                dosfiles = [
                    str(i)
                    for i in dosfiles
                    if "no_soc" not in str(i) and "spin_up" in str(i)
                ]
            elif get_SOC == False:
                dosfiles = [
                    str(i)
                    for i in dosfiles
                    if "no_soc" in str(i) and "spin_up" in str(i)
                ]

        if (self.active_SOC == True) and (self.spin == 2):
            if get_SOC == True:
                dosfiles = [
                    str(i)
                    for i in dosfiles
                    if "no_soc" not in str(i) and "spin_dn" in str(i)
                ]
            elif get_SOC == False:
                dosfiles = [
                    str(i)
                    for i in dosfiles
                    if "no_soc" in str(i) and "spin_dn" in str(i)
                ]

        elif (self.active_SOC == False) and (self.spin == 1):
            dosfiles = [str(i) for i in dosfiles if "spin_up" in str(i)]
        elif (self.active_SOC == False) and (self.spin == 2):
            dosfiles = [str(i) for i in dosfiles if "spin_dn" in str(i)]
        else:
            dosfiles = [str(i) for i in dosfiles]
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
        array = np.sum(list_of_arrays, axis=0)
        array[:, 0] /= len(list_of_arrays)
        return array

    def __get_total_dos(self, dos_per_atom):
        """ Sum over the total dos column of all atom DOS files."""
        total_dos = []
        for atom in dos_per_atom.keys():
            total_dos.append(dos_per_atom[atom][:, [0, 1]])
        total_dos = np.sum(total_dos, axis=0) / len(dos_per_atom.keys())
        return total_dos

    def __shift_to(self, energy, energy_reference):
        """ Shifts Fermi level of spectrum according to energy_reference attribute.

        If the work function has been calculated, the band structure will be shifted to the absolute value of the VBM.
        
        Returns:
            array: spectrum attribute
        """
        VBM = self.VBM - self.fermi_level
        CBM = self.CBM - self.fermi_level
        if self.work_function != None:
            energy -= self.upper_vacuum_potential
            VBM -= self.upper_vacuum_potential
            CBM -= self.upper_vacuum_potential
            energy_reference = "work_function"
        if (self.band_gap < 0.1) or (self.spin != None):
            energy_reference = "fermi"  # nothing to be done here
        elif (energy_reference == None) or (energy_reference == "none"):
            energy += self.fermi_level
            VBM += self.fermi_level
            CBM += self.fermi_level
        elif energy_reference == "middle":
            energy -= (VBM + CBM) / 2
            VBM = -self.band_gap / 2
            CBM = self.band_gap / 2
        elif energy_reference == "VBM":
            energy -= VBM
            CBM = self.band_gap
            VBM = 0
        elif energy_reference == "work_function":
            energy += self.fermi_level
            VBM += self.fermi_level
            CBM += self.fermi_level
            self.energy_reference = "work_function"
        return energy, VBM, CBM

    def plot(self, atom, orbital=None, fig=None, axes=None, fill="gradient", **kwargs):
        """ Plots the DOS of a single species.
            
        Args:
            atom (str): Species label.
            orbital (int): None --> total DOS; 0 --> s, 1 --> p, 2 --> d, 3 --> f etc.
            fig (matplotlib figure): Figure to draw the plot on.
            axes (matplotlib axes): Axes to draw the plot on.
            fill (str): Supported fill methods are None, "gradient", or "constant".
            mark_fermi_level (str): Color to mark fermi level, defaults to "none".
            **kwargs (dict): Passed to matplotlib plotting function.        
        
        Returns:
            axes: matplotlib axes object"""
        orbitals = {None: "", "s": 2, "p": 3, "d": 4, "f": 5}
        if fig == None:
            fig = plt.figure(figsize=(2, 4))
        if axes == None:
            axes = plt.gca()
        if orbital == None:  # This retrieves the total DOS.
            xy = self.dos_per_atom[atom][:, [0, 1]].copy()
        else:
            xy = self.dos_per_atom[atom][:, [0, orbitals[orbital]]].copy()
        x = xy[:, 1].copy()
        y = xy[:, 0].copy()
        for key in list(kwargs.keys()):
            if key in self._postprocess__global_plotproperties.keys():
                setattr(
                    self, key, kwargs.pop(key),
                )
            else:
                self._postprocess__mplkwargs[key] = kwargs[key]

        y, VBM, CBM = self.__shift_to(y, self.energy_reference)
        if self.fix_energy_limits == []:
            lower_ylimit = VBM - self.var_energy_limits
            upper_ylimit = CBM + self.var_energy_limits
        else:
            lower_ylimit = self.fix_energy_limits[0]
            upper_ylimit = self.fix_energy_limits[1]

        if fill == None:
            axes.plot(x, y, color=self.color, **self._postprocess__mplkwargs)
        elif fill == "gradient":
            axes.plot(x, y, color=self.color, alpha=0.9, **kwargs)
            self.__gradient_fill(x, y, axes, self.color)
        elif fill == "constant":
            axes.fill_betweenx(y, 0, x, color=self.color)
        xy = xy[(xy[:, 0] > lower_ylimit) & (xy[:, 0] < upper_ylimit)]
        axes.set_xlim([0, np.max(xy[:, 1]) + 0.1])
        axes.set_ylim([lower_ylimit, upper_ylimit])
        axes.set_xticks([])
        axes.set_xlabel("DOS")
        if self.energy_reference == None or self.energy_reference == "none":
            if self.mark_fermi_level != "none":
                axes.axhline(
                    y=(VBM + CBM) / 2,
                    color=self.mark_fermi_level,
                    alpha=0.5,
                    linestyle="--",
                )
            axes.set_ylabel("E [eV]")
        elif self.energy_reference == "work_function":
            if self.mark_fermi_level != "none":
                axes.axhline(
                    y=VBM, color=self.mark_fermi_level, alpha=0.5, linestyle="--"
                )
            axes.set_ylabel(r"E-E$_\mathrm{vacuum}$ [eV]")
        else:
            if self.mark_fermi_level != "none":
                axes.axhline(
                    y=0, color=self.mark_fermi_level, alpha=0.5, linestyle="--"
                )
            axes.set_ylabel(r"E-E$_\mathrm{F}$ [eV]")
        ylocs = ticker.MultipleLocator(
            base=0.5
        )  # this locator puts ticks at regular intervals
        axes.yaxis.set_major_locator(ylocs)
        axes.set_title(str(self.title), loc="center")
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
        from matplotlib.path import Path
        from matplotlib.patches import PathPatch

        z = np.empty((1, 100, 4), dtype=float)
        rgb = mcolors.colorConverter.to_rgb(color)
        z[:, :, :3] = rgb
        z[:, :, -1] = np.linspace(0, 1, 100)[None, :]
        xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
        im = axes.imshow(
            z, aspect="auto", extent=[xmin, xmax, ymin, ymax], origin="upper"
        )
        xy = np.column_stack([x, y])
        path = np.vstack([[0, ymin], xy, [0, ymax], [0, ymax], [0, ymax]])
        path = Path(path)
        patch = PathPatch(path, facecolor="none", edgecolor="none")
        # clip_path = Polygon(xy, facecolor="none", edgecolor="none", closed=False)
        axes.add_patch(patch)
        im.set_clip_path(patch)

    def plot_all_species(self, fig=None, axes=None, fill="gradient", **kwargs):
        """ Plot the DOS of all species colored by color_dict.
        
        Shares attributes with the plot() method.
        
        Args:
            fig (matplotlib figure): Figure to draw the plot on.
            axes (matplotlib axes): Axes to draw the plot on.
            fill (str): Supported fill methods are None, "gradient", or "constant".
            **kwargs (dict): Passed to matplotlib plotting function.       
        
        Returns:
            axes: matplotlib axes object"""
        if fig == None:
            fig = plt.figure(figsize=(2, 4))
        if axes != None:
            axes = plt.gca()
        else:
            axes = plt.subplot2grid((1, 1), (0, 0), fig=fig)
        handles = []
        xmax = []
        atoms = list(self.structure.species.keys())
        atoms.sort()
        for atom in atoms:
            color = self.color_dict[atom]
            axes = self.plot(atom, color=color, fig=fig, fill=fill, axes=axes, **kwargs)
            xmax.append(axes.get_xlim()[1])
            handles.append(Line2D([0], [0], color=color, label=atom, lw=1.0))
        axes.legend(handles=handles, frameon=True, loc="center right", fancybox=False)
        axes.set_xlim([0, max(xmax) * 1.05])
        return axes
