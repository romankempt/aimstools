import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.ticker as ticker

from scipy import interpolate
import ase.io, ase.cell

from AIMS_tools.misc import *
from AIMS_tools.postprocessing import postprocess


class bandstructure(postprocess):
    """ Band structure object.
    
    Contains all information about a single band structure instance, such as the energy spectrum, the band gap, Fermi level etc.

    Example:    
        >>> from AIMS_tools import bandstructure
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> bs = bandstructure.bandstructure("outputfile")
        >>> bs.plot()
        >>> plt.savefig("Name.png", dpi=300, transparent=False, bbox_inches="tight", facecolor="white")
        >>> plt.show()

    Args:    
        outputfile (str): Path to outputfile.
        get_SOC (bool): Retrieve spectrum with or without spin-orbit coupling (True/False), if calculated.
        spin (int): Retrieve spin channel 1 or 2. Defaults to None (spin-restricted) or 1 (collinear).
        shift_type (str): Shifts Fermi level. Options are None (default for metallic systems), "middle" for middle of band gap, and "VBM" for valence band maximum.
    
    Attributes:
        ksections (dict): Dictionary of path segments and corresponding band file.
        bandsegments (dict): Dictionary of path segments and corresponding eigenvalues.
        kpath (list): K-path labels following AFLOW conventions.
        kvectors (dict): Dictionary of k-point labels and fractional coordinates.
        klabel_coords (list): List of x-positions of the k-path labels.
        band_gap (float): Band gap energy in eV.
        spectrum (numpy array): Array of k-values and eigenvalues.
  
    """

    def __init__(self, outputfile, get_SOC=True, spin=None, shift_type="middle"):
        super().__init__(outputfile, get_SOC=get_SOC, spin=spin)
        if self.success == False:
            logging.critical("Calculation did not converge.")
            raise Exception()
        self.shift_type = shift_type
        self.bandfiles = self.__get_bandfiles(get_SOC)
        self.kpath = [i[0] for i in self.ksections]
        self.kpath += [self.ksections[-1][1]]  # retrieves the endpoint of the path
        self.ksections = dict(zip(self.ksections, self.bandfiles))
        self.bandsegments = self.__read_bandfiles()
        self.spectrum = self.__create_spectrum()

    def __str__(self):
        return "band structure"

    def properties(self):
        """ Prints out key properties of the band structure. """
        print("Sum Formula: {}".format(self.structure.atoms.get_chemical_formula()))
        print("Number of k-points for SCF: {}".format(self.k_points))
        cell = self.structure.atoms.cell
        if np.array_equal(cell[2], [0.0, 0.0, 100.0]):
            pbc = 2
            area = np.linalg.norm(np.cross(cell[0], cell[1]))
            kdens = self.k_points / area
        else:
            pbc = 3
            volume = self.structure.atoms.get_volume()
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

        print("Band gap: {:2f} eV (spin channel = {})".format(self.band_gap, self.spin))
        print(self.smallest_direct_gap)
        print("Path: ", self.kpath)
        import ase.spacegroup

        brav_latt = self.structure.atoms.cell.get_bravais_lattice(
            pbc=self.structure.atoms.pbc
        )
        sg = ase.spacegroup.get_spacegroup(self.structure.atoms, symprec=1e-2)
        print("Space group: {} (Nr. {}) \t precision = 1e-2".format(sg.symbol, sg.no))
        print("Bravais lattice: {}".format(brav_latt))

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
                self.path.joinpath(stem + "{:03d}.out".format(i + 1))
                for i in range(len(self.ksections))
            ]
        elif self.active_SOC == True and get_SOC == False:
            ### That's the ZORA case if SOC was calculated.
            bandfiles = [
                self.path.joinpath(stem + "{:03d}.out.no_soc".format(i + 1))
                for i in range(len(self.ksections))
            ]
        elif self.active_SOC == True and get_SOC == True:
            ### That's the SOC case if SOC was calculated.
            bandfiles = [
                self.path.joinpath(stem + "{:03d}.out".format(i + 1))
                for i in range(len(self.ksections))
            ]
        if (
            self.active_SOC == False and self.active_GW == True
        ):  ### That's the ZORA case with GW.
            bandfiles = [
                self.path.joinpath("GW_" + stem + "{:03d}.out".format(i + 1))
                for i in range(len(self.ksections))
            ]
        return bandfiles

    def __read_bandfiles(self):
        """ Reads in band.out files.
        
        Returns:
            dict : (kpoints, eigenvalues) tuple of ndarrays.
        
        Note:
            Automatically generates the reversed path sections.
        """
        bandsegments = {}
        for section, bandfile in self.ksections.items():
            try:
                array = [line.split() for line in open(bandfile)]
                array = np.array(array, dtype=float)
            except:
                logging.critical("File {} not found.".format(bandfile))
            kpoints = array[:, 1:4]
            eigenvalues = array[:, list(range(5, array.shape[1], 2))]
            array[:, 1:4] *= 2 * np.pi / (self.structure.atoms.cell.lengths() * bohr)
            bandsegments[section] = (kpoints, eigenvalues)
            ### Adding the reversed paths
            reverse = (section[1], section[0])
            if reverse not in self.ksections.keys():
                kpoints = kpoints[::-1]
                eigenvalues = eigenvalues[::-1]
                bandsegments[reverse] = (kpoints, eigenvalues)
        return bandsegments

    def __create_spectrum(self):
        """ Merges bandsegments to a single spectrum with suitable x-axis.
        
        Returns:
            ndarray : (nkpoints, nbands) array with first axis being the x-axis.
        """
        kstep = 0
        klabel_coords = [0.0]  # positions of x-ticks
        specs = []
        segments = [
            (self.kpath[i], self.kpath[i + 1]) for i in range(len(self.kpath) - 1)
        ]
        for segment in segments:
            kpoints = (
                self.bandsegments[segment][0]
                * 2
                * np.pi
                / (self.structure.atoms.cell.lengths() * bohr)
            )
            energies = self.bandsegments[segment][1]
            start = kstep
            kstep += np.linalg.norm(kpoints[-1] - kpoints[0])
            kaxis = np.linspace(start, kstep, kpoints.shape[0])
            klabel_coords.append(kstep)
            energies = np.insert(energies, 0, kaxis, axis=1)
            specs.append(energies)
        self.klabel_coords = klabel_coords
        spectrum = np.concatenate(specs, axis=0)
        VBM = np.max(spectrum[:, 1:][spectrum[:, 1:] < 0])
        CBM = np.min(spectrum[:, 1:][spectrum[:, 1:] > 0])
        self.band_gap = CBM - VBM
        return spectrum

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
                self.bandsegments[pair]
            except KeyError:
                print(
                    "The path {}-{} has not been calculated.".format(pair[0], pair[1])
                )
                break
        else:
            self.kpath = newpath
            self.spectrum = self.__create_spectrum()

    def __shift_to(self, energy):
        """ Shifts Fermi level of spectrum according to shift_type attribute.
        
        Returns:
            array: spectrum attribute
        """
        VBM = np.max(energy[energy < 0])
        CBM = np.min(energy[energy > 0])
        self.band_gap = CBM - VBM
        if (self.band_gap < 0.1) or (self.spin != None):
            self.shift_type = None
        if self.shift_type == None:
            energy += self.fermi_level
            self.shift_type = None
        elif self.shift_type == "middle":
            energy -= (VBM + CBM) / 2
        elif self.shift_type == "VBM":
            energy -= VBM
        return energy

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
            fig = plt.figure(figsize=(len(self.kpath) / 1.5, 3))
        if axes == None:
            axes = plt.gca()
        x = self.spectrum[:, 0]
        y = self.spectrum[:, 1:]
        y = self.__shift_to(y)
        VBM = np.max(y[y < 0]) if self.shift_type != None else self.fermi_level
        CBM = np.min(y[y > 0]) if self.shift_type != None else self.fermi_level
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
                xlabels.append(r"$\Gamma$")
            else:
                xlabels.append(self.kpath[i])
        axes.set_xticklabels(xlabels)
        ylocs = ticker.MultipleLocator(
            base=0.5
        )  # this locator puts ticks at regular intervals
        axes.yaxis.set_major_locator(ylocs)
        axes.set_xlabel("")
        if self.shift_type == None:
            axes.axhline(y=self.fermi_level, color="k", alpha=0.5, linestyle="--")
            axes.set_ylabel("E [eV]")
        else:
            axes.axhline(y=0, color="k", alpha=0.5, linestyle="--")
            axes.set_ylabel(r"E-E$_\mathrm{F}$ [eV]")
        axes.grid(which="major", axis="x", linestyle=":")
        axes.set_title(str(title), loc="center")
        return axes


class fatbandstructure(bandstructure):
    """ Band structure object. Inherits from bandstructure class. 

    Example:    
        >>> from AIMS_tools import bandstructure
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> bs = bandstructure.fatbandstructure("outputfile")
        >>> bs.plot_all_orbitals()
        >>> plt.savefig("Name.png", dpi=300, transparent=False, bbox_inches="tight", facecolor="white")
        >>> plt.show()        
    
    Args:
        filter_species (list): Only processes list of atom labels, e.g., ["W", "S"].
        readmode (bool): To be implemented.
    
    Attributes:
        mlk_bandsegments (dict): Nested dictionary of path segments and ndarrays containing data.
        atom_contributions (dict): Nested dictionary of dictionaries containing {atom : {section : {kvalues, contributions}}}.
        atom_spectra (dict): Nested dictionary of atom index keys and data for plotting aligned with the kpath.
        atoms_to_plot (dict): Pairs of index and chemical symbols for atoms to plot.
 
    """

    def __init__(
        self,
        outputfile,
        get_SOC=True,
        spin=None,
        shift_type="middle",
        filter_species=[],
    ):
        super().__init__(outputfile, get_SOC=get_SOC, shift_type=shift_type, spin=spin)
        # get_SOC is true because for mulliken bands, both spin channels are written to the same file.
        self.filter_species = (
            list([str(i) for i in self.structure.atom_indices.values()])
            if filter_species == []
            else filter_species
        )
        self.atoms_to_plot = {
            k: v
            for k, v in self.structure.atom_indices.items()
            if v in self.filter_species
        }
        self.contributions = {}
        self.mlk_bandfiles = self.__get_mlk_bandfiles(get_SOC)
        self.ksections = dict(
            zip(list(self.ksections.keys()), list(self.mlk_bandfiles))
        )
        if not self.path.joinpath("fatbands_atom_contributions.zip").exists():
            self.mlk_bandsegments = self.__read_mlk_bandfiles()
            self.atom_contributions = self.__collect_contributions()
            self.atom_spectra = self.__create_spectra()
            start = time.time()
            self.__write_contributions()
            end = time.time()
            duration = end - start
            logging.info(
                "Contributions were written to fatbands_atom_contributions.zip in {: .4f} s.".format(
                    duration
                )
            )
        elif self.path.joinpath("fatbands_atom_contributions.zip").exists():
            logging.info(
                "Reading contributions from fatbands_atom_contributions.zip ..."
            )
            self.atom_contributions = self.__read_contributions()
            self.atom_spectra = self.__create_spectra()

    def __str__(self):
        return "fat band structure"

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
                self.path.joinpath(stem + "{:03d}.out".format(i + 1))
                for i in range(len(self.ksections))
            ]
        elif self.active_SOC == True and get_SOC == True:
            bandfiles = [
                self.path.joinpath(stem + "{:03d}.out".format(i + 1))
                for i in range(len(self.ksections))
            ]
        return bandfiles

    def __read_mlk_bandfile(self, bandfile):
        """ Reads in mlk bandfile.

        Args:
            bandfile (str): Path to bandfile.
        
        Attributes:
            nkpoints (dict): Number of k-points.
            nstates (int): Number of states.
            natoms (int): Number of atoms.

        Returns:
            ndarray: shape (nkpoints, nstates*natoms, nvalues), where the
            values axis has the following structure:
            [index, eigenvalue, occupation, atom, spin, total, s, p, d, f, g]

        """
        from itertools import groupby, zip_longest

        try:
            with open(bandfile, "r") as file:
                content = [line.strip() for line in file.readlines()]
            assert content != None, "Could not read data from {}".format(bandfile)
        except:
            return "File {} not found.".format(bandfile)
        ## filter content
        content = [line for line in content if "State" not in line]
        kpoints = [k for k in content if "k point number" in k]
        kpoints = np.array(
            [
                k.replace(":", "")
                .replace("(", "")
                .replace(")", "")
                .strip("k point number")
                .strip()
                .split()
                for k in kpoints
            ],
            dtype=float,
        )
        # kpoints (index, kx, ky, kz) ndarray
        values = [
            list(group)
            for k, group in groupby(content, lambda x: "k point number" in x)
            if not k
        ]
        # This fills missing entries accurately
        for kpoint in range(len(values)):
            entries = [i.split() for i in values[kpoint]]
            entries = np.array(list(zip_longest(*entries, fillvalue=0)), dtype=float).T
            values[kpoint] = entries

        values = np.array(values, dtype=float)
        nkpoints = values.shape[0]
        assert nkpoints == len(kpoints), "Number of k-points does not match."
        self.natoms = len(self.structure.atoms)
        self.nstates = int(values.shape[1] / self.natoms)
        if (self.spin == None) and (self.active_SOC == False):
            values = np.insert(values, 4, [1] * self.nstates * self.natoms, axis=2)
        self.ncons = values.shape[2]
        return kpoints, values

    def __read_mlk_bandfiles(self):
        """ Iterates __read_mlk_bandfile() over bandfiles.

        Returns:
            dict: Nested dictionary of bandfile : dict pairs, where dict
            contains the pairs kpoints : ndarray and values : ndarray.
        """
        logging.info("Reading in mulliken band files in serial...")
        self.nkpoints = {}
        bandsegments = {}
        for section, bandfile in self.ksections.items():
            start = time.time()
            kpoints, eigenvalues = self.__read_mlk_bandfile(bandfile)
            self.nkpoints[section] = kpoints.shape[0]
            end = time.time()
            duration = end - start
            logging.info("\t Processed {} in {:.2f} s.".format(bandfile.name, duration))
            segment = (kpoints, eigenvalues)
            bandsegments[section] = segment
            ### Adding the reversed paths
            reverse = (section[1], section[0])
            if reverse not in self.ksections.keys():
                eigenvalues = eigenvalues[::-1, :, :]
                segment = (kpoints, eigenvalues)
                bandsegments[reverse] = segment
        return bandsegments

    def __collect_atom_contributions(self, atom):
        """ Collects energies, k-axis and contributions per atom.
        
        Args:
            atom (int): Index of atom in self.atoms_to_plot.
        
        Returns:
            dict : Keys are path section tuples (e.g. (G, Y)) values are (kaxis, values) tuples of ndarrays with shapes (nkpoints,) and (nkpoints, nstates, ncons)
        
        """
        segments = {}
        for section, values in self.mlk_bandsegments.items():
            kpoints = (
                values[0][:, 1:]
                * 2
                * np.pi
                / (self.structure.atoms.cell.lengths() * bohr)
            )
            ev = values[1]
            kstep = np.linalg.norm(kpoints[-1] - kpoints[0])
            kaxis = np.linspace(0, np.max(kstep), kpoints.shape[0], endpoint=False)
            # Filter by atom:
            ev = ev[ev[:, :, 3] == atom].reshape(
                kpoints.shape[0], self.nstates, self.ncons
            )
            if self.spin != None:
                # Filter by spin channel:
                ev = ev[ev[:, :, 4] == self.spin].reshape(
                    kpoints.shape[0], int(self.nstates / 2), self.ncons
                )
            segments[section] = (kaxis, ev)
        return segments

    def __collect_contributions(self):
        """ Iterates __collect_atom_contributions over atoms.
        
        Returns:
            dict: Nested dictionary of {atom index : ( ksection : (kaxis, values))}.
         """
        atom_contributions = {}
        for atom in self.atoms_to_plot.keys():
            atom_contributions[atom] = self.__collect_atom_contributions(atom)
        return atom_contributions

    def __create_spectra(self):
        """ Concatenates contributions and k-axis according to k-path.
        
        Returns:
            dict: Dictionary of atom : (kaxis, spectrum) pairs.
        """
        segments = [
            (self.kpath[i], self.kpath[i + 1]) for i in range(len(self.kpath) - 1)
        ]
        nkpoints_per_sec = self.nkpoints
        reverse = {(k[1], k[0]): v for (k, v) in nkpoints_per_sec.items()}
        nkpoints_per_sec.update(reverse)
        nkpoints = sum([v for k, v in nkpoints_per_sec.items() if k in segments])
        channels = 1 if self.spin == None else 2
        atom_spectrum = {}
        for atom in self.atoms_to_plot.keys():
            klabel_coords = [0.0]
            start_index = 0
            kaxis = np.zeros((nkpoints))
            spectrum = np.zeros((nkpoints, int(self.nstates / channels), self.ncons))
            for section in segments:
                klength = self.atom_contributions[atom][section][0] + klabel_coords[-1]
                klabel_coords.append(klength[-1])
                end_index = start_index + klength.shape[0]
                kaxis[start_index:end_index] = klength
                cons = self.atom_contributions[atom][section][1]
                spectrum[start_index:end_index, :, :] = cons
                start_index = end_index
            atom_spectrum[atom] = (kaxis, spectrum)
            self.klabel_coords = klabel_coords
        energy = atom_spectrum[1][1][:, :, 1]
        self.band_gap = np.abs(np.min(energy[energy > 0]) - np.max(energy[energy < 0]))
        return atom_spectrum

    def sum_all_species_contributions(self):
        """ Sums (normalized) atomic contributions for the same species.

        Modifies atom_spectra attribute and reduces atoms_to_plot attribute.

        """
        logging.info("Summing up contributions of same species ...")
        atoms = self.atoms_to_plot
        reverse_atoms = {}
        for key, value in atoms.items():
            reverse_atoms.setdefault(value, set()).add(key)
        duplicates = [
            values for key, values in reverse_atoms.items() if len(values) > 1
        ]
        for duplicate in duplicates:
            number = len(duplicate)
            new_key = min(list(duplicate))
            sum_contribution = np.zeros(self.atom_spectra[new_key][1].shape)
            for key in duplicate:
                kaxis = self.atom_spectra[key][0]
                sum_contribution += self.atom_spectra.pop(key)[1]
                if key != new_key:
                    self.atoms_to_plot.pop(key)
            # Contributions are normalized, energies not:
            sum_contribution[:, :, [0, 1, 2, 3, 4]] /= number
            self.atom_spectra[new_key] = (kaxis, sum_contribution)

    def sum_contributions(self, list_of_species):
        """ Sums contributions of species. 
        
        Modifies both atom_indices and atom_spectra attribute.

        Args:
            list_of_species (list): List of labels (["H", "C", ...]) or list of indices ([1, 2, 3, ...]) or mix of both is accepted.
        
        Note:
            Might give unexpected results in conjunction with custom_path. A custom k-path should be specified first before calling this option.
        
        """
        assert type(list_of_species) == list
        for entry in range(len(list_of_species)):
            species = list_of_species[entry]
            if type(species) == int:
                assert species in list(
                    self.atoms_to_plot.keys()
                ), "Index {} not in Atoms!".format(species)
            elif type(species) == str:
                assert species in list(
                    self.atoms_to_plot.values()
                ), "Species {} not in Atoms!".format(species)
                list_of_species[entry] = [
                    k for k, v in self.atoms_to_plot.items() if v == species
                ][0]
            else:
                logging.error("Format type not recognised.")

        sum_cons = np.zeros(self.atom_spectra[list_of_species[0]][1].shape)
        label = ""
        for entry in list_of_species:
            label += self.atoms_to_plot.pop(entry)
            kaxis = self.atom_spectra[entry][0]
            sum_cons += self.atom_spectra.pop(entry)[1]
        sum_cons[:, :, [0, 1, 2, 3, 4]] /= len(list_of_species)
        new_index = min(list_of_species)
        self.atoms_to_plot[new_index] = label
        self.atom_spectra[new_index] = (kaxis, sum_cons)

    def __write_contributions(self):
        from zipfile import ZipFile

        files = []
        for index, atom in self.atoms_to_plot.items():
            for segment in self.mlk_bandsegments.keys():
                name1 = "{}_{}_{}-{}_{}_fatband_kaxis.npy".format(
                    atom, index, segment[0], segment[1], self.nkpoints[segment]
                )
                np.save(
                    name1, self.atom_contributions[index][segment][0],
                )
                name2 = "{}_{}_{}-{}_{}_fatband_contribution.npy".format(
                    atom, index, segment[0], segment[1], self.nkpoints[segment]
                )
                np.save(
                    name2, self.atom_contributions[index][segment][1],
                )
                files.append(name1)
                files.append(name2)
        with ZipFile(
            str(self.path.joinpath("fatbands_atom_contributions.zip")), "w"
        ) as zipObj:
            for n in files:
                zipObj.write(n)
        for n in files:
            os.remove(n)

    def __read_contributions(self):
        import zipfile

        atom_contributions = {}
        nkpoints = {}
        with zipfile.ZipFile(
            self.path.joinpath("fatbands_atom_contributions.zip")
        ) as zipref:
            zipref.extractall(self.path)
        kfiles = self.path.glob("*fatband*kaxis*.npy")
        confiles = self.path.glob("*fatband*contribution*.npy")
        for j, k in zip(kfiles, confiles):
            atom, index, segment, nk1 = str(j).split("_")[:4]
            segment = (segment.split("-")[0], segment.split("-")[1])
            at2, ind2, seg2, nk2 = str(k).split("_")[:4]
            seg2 = (seg2.split("-")[0], seg2.split("-")[1])
            assert atom == at2, "Atom not matching."
            assert index == ind2, "Index not matching."
            assert segment == seg2, "Segment not matching."
            assert nk1 == nk2, "Nkpoints not matching."
            nk1 = int(nk1)
            kaxis = np.load(str(j))
            cons = np.load(str(k))
            self.natoms = len(self.structure.atoms)
            self.nstates = int(cons.shape[1])
            self.ncons = int(cons.shape[2])
            nkpoints[segment] = nk1
            index = int(index)
            if index not in atom_contributions.keys():
                atom_contributions[index] = {segment: (kaxis, cons)}
            else:
                atom_contributions[index].update({segment: (kaxis, cons)})
            os.remove(str(j))
            os.remove(str(k))
        self.nkpoints = nkpoints
        return atom_contributions

    def sort_atoms(self):
        """ Sorts by heaviest atom. 
        
        Changes atoms_to_plot attribute.
        """
        logging.info("Sorting by heaviest atom...")
        keys = list(self.atoms_to_plot.keys())
        vals = list(self.atoms_to_plot.values())
        pse_vals = [pse[val] for val in vals]
        sorted_keys = [
            x for _, x in sorted(zip(pse_vals, keys), key=lambda pair: pair[0])
        ]
        self.atoms_to_plot = {key: self.atoms_to_plot[key] for key in sorted_keys}

    def custom_path(self, custompath):
        """ This function takes in a custom path of form K1-K2-K3 for plotting.

        Changes kpath attribute according to custompath and recreates atom_spectra according to new kpath.

        Args:
            custompath (str): Hyphen-separated string of path labels, e.g., "G-M-X".
        Note:
            Only the paths that have been calculated in the control.in can be plotted.
         """
        newpath = custompath.split("-")
        check = [(newpath[i], newpath[i + 1]) for i in range(len(newpath) - 1)]
        for pair in check:
            try:
                self.mlk_bandsegments[pair]
            except KeyError:
                print(
                    "The path {}-{} has not been calculated.".format(pair[0], pair[1])
                )
                break
        else:
            self.kpath = newpath
            self.atoms_to_plot = {
                k: v
                for k, v in self.structure.atom_indices.items()
                if v in self.filter_species
            }
            self.atom_spectra = self.__create_spectra()

    def __shift_to(self, energy):
        VBM = np.max(energy[energy[:, :] < 0])
        CBM = np.min(energy[energy[:, :] > 0])
        self.band_gap = CBM - VBM
        if self.band_gap < 0.1:
            shift_type = None
        elif self.shift_type == "middle":
            energy[:, :] -= (VBM + CBM) / 2
        elif self.shift_type == "VBM":
            energy[:, :] -= VBM
        return energy

    def plot(
        self,
        title="",
        fig=None,
        axes=None,
        color="k",
        var_energy_limits=1.0,
        fix_energy_limits=[],
        mark_gap="lightgray",
        kwargs={"alpha": 0.25, "linewidth": 0.5},
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
            fig = plt.figure(figsize=(len(self.kpath) / 1.5, 3))
        if axes == None:
            axes = plt.gca()
        x = self.atom_spectra[1][0]
        y = self.atom_spectra[1][1][:, :, 1]
        y = self.__shift_to(y)
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
                xlabels.append(r"$\Gamma$")
            else:
                xlabels.append(self.kpath[i])
        axes.set_xticklabels(xlabels)
        axes.set_ylabel(r"E-E$_\mathrm{F}$ [eV]")
        ylocs = ticker.MultipleLocator(
            base=0.5
        )  # this locator puts ticks at regular intervals
        axes.yaxis.set_major_locator(ylocs)
        axes.set_xlabel("")
        axes.axhline(y=0, color="k", alpha=0.5, linestyle="--")
        axes.grid(which="major", axis="x", linestyle=":")
        axes.set_title(str(title), loc="center")
        return axes

    def plot_mlk(
        self,
        atom,
        contribution,
        mode="lines",
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
            atom (int, str): Index or label of atom in self.atoms_to_plot.
            contribution (str): Spectral contribution tot, s, p, d or f or user-defined ndarray.
            mode (str): "lines" or "scatter". Defaults to "lines".
            title (str): Title of the plot.
            fig (matplotlib figure): Figure to draw the plot on.
            axes (matplotlib axes): Axes to draw the plot on.
            cmap (str): Matplotlib colormap instance (e.g., "Blues", "Oranges", "Purples", "Reds", "Greens").
            var_energy_limits (int): Variable energy range above and below the band gap to show.
            fix_energy_limits (list): List of lower and upper energy limit to show.
            nbands (int): False or integer. Number of bands above and below the Fermi level to colorize.
            interpolation_step (float): False or float. Performs 1D interpolation of every band with the specified
            step size (e.g., 0.001). May cause substantial computational effort.
            **kwargs (dict): Currently not used.
        
        Returns:
            axes: matplotlib axes object"""
        if fig == None:
            fig = plt.figure(figsize=(len(self.kpath) / 1.5, 3))
        if axes == None:
            axes = plt.gca()
        con_dict = {"tot": 5, "s": 6, "p": 7, "d": 8, "f": 9}

        if type(atom) == str:
            atom = [k for k, v in self.atoms_to_plot.items() if v == atom][0]

        x = self.atom_spectra[atom][0]
        y = self.atom_spectra[atom][1][:, :, 1]  # energy
        y = self.__shift_to(y)
        if type(contribution) == str:
            con = con_dict[contribution]
            con = self.atom_spectra[atom][1][:, :, con]
        else:
            con = contribution

        VBM = np.max(y[y < 0])
        CBM = np.min(y[y > 0])

        # adjusting y limits
        if fix_energy_limits == []:
            lower_ylimit = VBM - var_energy_limits
            upper_ylimit = CBM + var_energy_limits
        else:
            lower_ylimit = fix_energy_limits[0]
            upper_ylimit = fix_energy_limits[1]

        # cutting nbands
        if nbands != False:
            index = np.where(y == VBM)
            col = index[1][0]
            y = y[:, col - nbands : col + nbands + 1]
            con = con[:, col - nbands : col + nbands + 1]

        # defining the color map
        cmap = plt.get_cmap(cmap)
        my_cmap = cmap(np.arange(cmap.N))
        my_cmap[:, -1] = np.linspace(0, 1, cmap.N)  # this adds alpha
        my_cmap = ListedColormap(my_cmap)

        for band in range(y.shape[1]):
            band_x = x
            band_y = y[:, band]
            band_width = con[:, band]

            if interpolation_step != False:
                f1 = interpolate.interp1d(x, band_y)
                f2 = interpolate.interp1d(x, band_width)
                band_x = np.arange(0, np.max(x), interpolation_step)
                band_y = f1(band_x)
                band_width = f2(band_x)

            if mode == "lines":
                band_width = band_width[:-1]
                points = np.array([band_x, band_y]).T.reshape(-1, 1, 2)
                segments = np.concatenate(
                    [points[:-1], points[1:]], axis=1
                )  # this reshapes it into (x1, x2) (y1, y2) pairs
                lc = LineCollection(
                    segments,
                    linewidths=1.5,  # band_width * 2.5,  # making them fat
                    cmap=my_cmap,
                    norm=plt.Normalize(0, 1),
                    capstyle="round",
                )
                lc.set_array(band_width)
                axes.add_collection(lc)
            elif mode == "scatter":
                axes.scatter(
                    band_x,
                    band_y,
                    c=band_width,
                    cmap=my_cmap,
                    norm=plt.Normalize(0, 1),
                    s=(band_width * 2),
                )

        axes.set_ylim([lower_ylimit, upper_ylimit])
        axes.set_xlim([0, np.max(x)])
        axes.set_xticks(self.klabel_coords)
        xlabels = []
        for i in range(len(self.kpath)):
            if self.kpath[i] == "G":
                xlabels.append(r"$\Gamma$")
            else:
                xlabels.append(self.kpath[i])
        axes.set_xticklabels(xlabels)
        axes.set_ylabel(r"E-E$_\mathrm{F}$ [eV]")
        ylocs = ticker.MultipleLocator(
            base=0.5
        )  # this locator puts ticks at regular intervals
        axes.yaxis.set_major_locator(ylocs)
        axes.set_xlabel("")
        axes.axhline(y=0, color="k", alpha=0.5, linestyle="--")
        axes.grid(which="major", axis="x", linestyle=":")
        axes.set_title(str(title), loc="center")
        return axes

    def plot_all_species(
        self,
        axes=None,
        fig=None,
        mode="lines",
        title="",
        sum=True,
        var_energy_limits=1.0,
        fix_energy_limits=[],
        nbands=False,
        interpolation_step=False,
        kwargs={"alpha": 0.15, "linewidth": 0.5},
    ):
        """ Plots a fatbandstructure instance with all species overlaid.
        
        Shares attributes with the in-built plot function. Gives an error if invoked with more than 5 species.
      
        Args:
            sum (bool): True or False. Sums contributions of same species.
            **kwargs (dict): Keyword arguments are passed to the plot() method.
        
        Returns:
            axes: matplotlib axes object"""
        if fig == None:
            fig = plt.figure(figsize=(len(self.kpath) / 1.5, 4))
        if axes != None:
            axes = plt.gca()
        else:
            axes = plt.subplot2grid((1, 1), (0, 0), fig=fig)
        # plotting background energy
        axes = self.plot(
            fig=fig, axes=axes, color="lightgray", mark_gap=False, kwargs=kwargs
        )

        if sum == True:
            self.sum_all_species_contributions()
            self.sort_atoms()
        if len(self.atoms_to_plot.keys()) > 5:
            logging.error(
                """Humans can't perceive enough colors to make a band structure plot
            possible with {} species.""".format(
                    len(self.atoms_to_plot.keys())
                )
            )
            return None
        if interpolation_step != False:
            logging.info(
                """Performing 1D interpolation of every single band to obtain a smoother plot."""
            )
        cmaps = ["Blues", "Oranges", "Greens", "Purples", "Reds"]
        colors = ["darkblue", "darkorange", "darkgreen", "darkviolet", "darkred"]
        handles = []
        i = 0
        for atom, label in self.atoms_to_plot.items():
            self.plot_mlk(
                atom,
                "tot",
                mode=mode,
                cmap=cmaps[i],
                axes=axes,
                fig=fig,
                title=title,
                var_energy_limits=var_energy_limits,
                fix_energy_limits=fix_energy_limits,
                nbands=nbands,
                interpolation_step=interpolation_step,
            )
            handles.append(Line2D([0], [0], color=colors[i], label=label, lw=1.5))
            i += 1
        lgd = axes.legend(
            handles=handles,
            frameon=True,
            fancybox=False,
            borderpad=0.4,
            loc="upper right",
            # bbox_to_anchor=(1, 1),
        )
        return axes

    def sum_orbitals(self):
        """ Sums orbital contributions of all species.
        
        Returns:
            ndarray : shape (nkpoints, nstates, [index, eigenvalue, occupation, atom, spin, total, s, p, d, f, g])
         """
        orbital_contributions = np.zeros(self.atom_spectra[1][1].shape)
        for index, species in self.atoms_to_plot.items():
            orbital_contributions += self.atom_spectra[index][1]
        orbital_contributions[:, :, [0, 1, 2, 3, 4]] /= len(
            list(self.atoms_to_plot.keys())
        )
        return orbital_contributions

    def plot_all_orbitals(
        self,
        axes=None,
        fig=None,
        mode="lines",
        title="",
        var_energy_limits=1.0,
        fix_energy_limits=[],
        nbands=False,
        interpolation_step=False,
        kwargs={"alpha": 0.15, "linewidth": 0.5},
    ):
        """ Plots a fatbandstructure instance with all orbital characters overlaid.
        
        Shares attributes with the in-built plot function.

        Args:
            **kwargs (dict): Keyword arguments are passed to the plot() method.
                
        Returns:
            axes: matplotlib axes object """
        if fig == None:
            fig = plt.figure(figsize=(len(self.kpath) / 1.5, 4))
        if axes != None:
            axes = plt.gca()
        else:
            axes = plt.subplot2grid((1, 1), (0, 0), fig=fig)

        # plotting background energy
        axes = self.plot(
            fig=fig, axes=axes, color="lightgray", mark_gap=False, kwargs=kwargs
        )

        if interpolation_step != False:
            logging.info(
                """Performing 1D interpolation of every single band to obtain a smoother plot."""
            )

        cmaps = ["Oranges", "Blues", "Greens", "Purples", "Reds"]
        colors = ["darkorange", "darkblue", "darkgreen", "darkviolet", "darkred"]
        handles = []
        i = 0
        orbitals = self.sum_orbitals()
        for orbital in [6, 7, 8]:
            self.plot_mlk(
                1,
                orbitals[:, :, orbital],
                mode=mode,
                cmap=cmaps[i],
                axes=axes,
                fig=fig,
                title=title,
                var_energy_limits=var_energy_limits,
                fix_energy_limits=fix_energy_limits,
                nbands=nbands,
                interpolation_step=interpolation_step,
            )
            labels = ["s", "p", "d"]
            handles.append(Line2D([0], [0], color=colors[i], label=labels[i], lw=1.5))
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

