<<<<<<< HEAD
import numpy as np
import glob, sys, os, shutil
import matplotlib.pyplot as plt
from pathlib import Path as Path
from scipy import interpolate
from AIMS_tools import bandstructure
import ase.io, ase.cell, ase.spacegroup
import math
from scipy.optimize import curve_fit
import scipy.constants as const


plt.style.use("seaborn-ticks")

### units
hartree_to_eV = 27.211402666173235  # eV to hartree


class eff_mass(bandstructure.bandstructure):
    """ The effective mass calculator inherits from the bandstructure class.
    The calculation has several steps:
    - The k-point coordinates of the VBM and CBM are determined numerically.
    - Sampling paths around these two points are prepared. These consider dimensionality.
    - If needed, the files for a new band structure calculation are prepared. If possible, these
      can be simply restarted as a non-self-consistent calculation, if the keyword
      elsi_restart read_and_write freq was included in the first calculation.
    - The lowest conduction band and highest valence band are fitted with parabolic functions
      for all directions in k-space to evaluate the effective mass components.
    - Quantities derived from the effective mass components are calculated including degeneracy factors.
    """

    def __init__(self, outputfile, get_SOC=True, custom_path="", shift_to="middle"):
        super().__init__(
            outputfile, get_SOC=get_SOC, custom_path=custom_path, shift_to=shift_to
        )
        if custom_path != "":
            self.custom_path(custom_path)

        self.energies = (
            self.spectrum[:, 1:] / hartree_to_eV
        )  # converting to atomic units
        self.sg = ase.spacegroup.get_spacegroup(self.cell)

        ### Execute first steps, if calculation has not been run yet
        if not self.path.joinpath("eff_mass").exists() and "eff_mass" != str(
            self.path.parts[-1]
        ):
            self.calc_dir = self.path.joinpath("eff_mass")
            print(
                "Preparing files for calculation in directory {}.".format(
                    str(self.calc_dir)
                )
            )
            self.VBM = np.max(self.energies[self.energies < 0])
            self.CBM = np.min(self.energies[self.energies > 0])
            self.VBM_locs, self.CBM_locs = (  # list of locations with the same energy
                self.get_kcoords_of(self.VBM),
                self.get_kcoords_of(self.CBM),
            )
            self.sample_bands = {}
            self.resample(self.VBM_locs, "VBM")
            self.resample(self.CBM_locs, "CBM")
            self.prepare_new_calc()
            sys.exit()

        ### Execute if calculation has been run
        if str(self.path.parts[-1]) == "eff_mass":
            print("Evaluating calculations in directory {}.".format(str(self.path)))
            self.masses = self.band_fitting_method(0)
            # self.evaluate_masses(self.masses)
            # if self.active_SOC == True:
            #     self.SOC_masses = self.band_fitting_method(1)

    #                self.evaluate_masses(self.SOC_masses)

    ###   Methods
    def band_fitting_method(self, band_index):
        directions = {}
        ar = np.zeros((int(self.spectrum.shape[0] / 2), 7))
        j = 0
        t = 0
        for axis in list(self.ksections.keys()):
            if "VBM" in axis[0]:  # should be 6 directions
                self.custom_path(axis[0] + "-" + axis[1])
                x = self.spectrum[:, 0]
                y = self.spectrum[:, 1:] / hartree_to_eV
                VBM = np.max(y[y < 0])
                index = np.where(y == VBM)
                loc = index[0][0]
                band = index[1][0]
                band -= band_index
                VBM = y[loc, band]  # this realigns the energy to off-split bands
                end = t + len(x)
                ar[t:end, j] = x
                ar[t:end, 6] = y[:, band]
                j += 1
                t += len(x)

        def tensor(Vars, E0, mxx, myy, mzz, mxy, mxz, myz):
            xx, yy, zz, xy, xz, yz = (
                Vars[:, 0],
                Vars[:, 1],
                Vars[:, 2],
                Vars[:, 3],
                Vars[:, 4],
                Vars[:, 5],
            )
            return (
                E0
                + 1 / (2 * mxx) * xx ** 2
                + 1 / (2 * myy) * yy ** 2
                + 1 / (2 * mzz) * zz ** 2
                + 1 / (2 * mxy) * xy ** 2
                + 1 / (2 * mxz) * xz ** 2
                + 1 / (2 * myz) * yz ** 2
            )

        p0 = VBM, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1
        bounds = (
            (VBM - 0.00001, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf),
            (VBM + 0.00001, -0.0001, -0.0001, -0.0001, -0.0001, -0.0001, -0.0001),
        )
        popt, pcov = curve_fit(tensor, ar[:, 0:6], ar[:, 6], bounds=bounds)
        err = np.sqrt(np.diag(pcov))[1]

        mtens = np.array(
            [
                [popt[1], popt[4], popt[5]],
                [popt[4], popt[2], popt[6]],
                [popt[5], popt[6], popt[3]],
            ]
        )
        # mtens = np.linalg.inv(mtens)
        w, v = np.linalg.eig(mtens)
        print(mtens)
        print(w)
        print(v / self.rec_cell_lengths)
        # _, direction, label = axis[0].split("_")
        # kpoint = tuple((self.kvectors[axis[0]] + self.kvectors[axis[1]]) / 2)
        # degeneracy = len(self.sg.equivalent_lattice_points(kpoint))
        # if label not in masses.keys():
        #     masses[label] = {
        #         "kpoint": kpoint,
        #         direction: popt[1],
        #         "degeneracy": degeneracy,
        #     }
        # else:
        #     masses[label].update(
        #         {"kpoint": kpoint, direction: popt[1], "degeneracy": degeneracy}
        #     )

    def old_band_fitting_method(self, band_index):
        """ Fits parabolic functions for each resampled extreme point
        in x, y and z direction. For each fitting, a plot is saved including
        the fitted mass, the fitting error, and the k-point. The points are labelled as
        VBM or CBM and the band index, where +1 or -1 refers to to the band below or above
        (which is the case for SOC calculations).
        
        band_index : int (1 to obtain VBM-1 and CBM+1)
        """
        masses = {}
        for axis in list(self.ksections.keys()):
            self.custom_path(axis[0] + "-" + axis[1])
            x = self.spectrum[:, 0]
            y = self.spectrum[:, 1:] / hartree_to_eV
            _, direction, label = axis[0].split("_")
            kpoint = tuple((self.kvectors[axis[0]] + self.kvectors[axis[1]]) / 2)
            degeneracy = len(self.sg.equivalent_lattice_points(kpoint))
            if "VBM" in axis[0]:
                VBM = np.max(y[y < 0])
                index = np.where(y == VBM)
                loc = index[0][0]
                band = index[1][0]
                band -= band_index
                VBM = y[loc, band]  # this realigns the energy to off-split bands
                p0 = VBM, -0.1
                bounds = ((VBM - 0.00001, -10), (VBM + 0.00001, 10))
                label = label + "-" + str(band_index)
            if "CBM" in axis[0]:
                CBM = np.min(y[y > 0])
                index = np.where(y == CBM)
                loc = index[0][0]
                band = index[1][0]
                band += band_index
                CBM = y[loc, band]  # this realigns the energy to off-split bands
                p0 = CBM, 0.1
                bounds = ((CBM - 0.00001, -10), (CBM + 0.00001, 10))
                label = label + "+" + str(band_index)
            x = x - x[loc]
            y = y[:, band]

            def parabola(k, E0, m):
                return E0 + ((k ** 2) / (2 * m))

            popt, pcov = curve_fit(parabola, x, y, p0, bounds=bounds)
            err = np.sqrt(np.diag(pcov))[1]

            if label not in masses.keys():
                masses[label] = {
                    "kpoint": kpoint,
                    direction: popt[1],
                    "degeneracy": degeneracy,
                }
            else:
                masses[label].update(
                    {"kpoint": kpoint, direction: popt[1], "degeneracy": degeneracy}
                )
            plt.figure(figsize=(2, 3))
            plt.scatter(x, y, color="crimson", alpha=0.7, s=10)
            plt.plot(x, parabola(x, *popt), color="blue")
            plt.plot(x, self.spectrum[:, 1:] / hartree_to_eV, color="gray", alpha=0.5)
            plt.xlim([np.min(x), np.max(x)])
            plt.ylim([np.min(y) - 0.001, np.max(y) + 0.001])
            plt.xlabel("$\Delta$k$_" + direction + "$ [2$\pi$/bohr]")
            plt.ylabel("E-E$_F$ [Hartree]")
            if "VBM" in label:
                plt.title("VBM -" + str(band_index))
                plt.annotate(
                    """k = ({:.2f} {:.2f} {:.2f}) \nm = {:.2f}$\pm${:.2f}""".format(
                        *kpoint, popt[1], err
                    ),
                    xy=(np.min(x) + 0.015, np.min(y) - 0.0005),
                )
                os.chdir(str(self.path))
                plt.savefig(
                    "VBM_-" + str(band_index) + direction + ".png",
                    dpi=300,
                    bbox_inches="tight",
                    faceolor="white",
                    transparent=False,
                )
            if "CBM" in label:
                plt.title("CBM +" + str(band_index))
                plt.annotate(
                    """k = ({:.2f} {:.2f} {:.2f})\nm = {:.2f}$\pm${:.2f}""".format(
                        *kpoint, popt[1], err
                    ),
                    xy=(np.min(x) + 0.015, np.max(y)),
                )
                os.chdir(str(self.path))
                plt.savefig(
                    "CBM_+" + str(band_index) + direction + ".png",
                    dpi=300,
                    bbox_inches="tight",
                    faceolor="white",
                    transparent=False,
                )
        return masses

    def evaluate_masses(self, massdict):
        print("Calculating mass-component dervied quantities.")
        for point in massdict.keys():
            masses = []
            directions = [
                i for i in list(massdict[point].keys()) if i in ["x", "y", "z"]
            ]
            for i in directions:
                masses.append(np.abs(massdict[point][i]))
            DOSmass = (massdict[point]["degeneracy"] ** 2 * np.prod(masses)) ** (
                1 / len(masses)
            )
            print(
                """DOS mass of {} at K-Point {:.2f} {:.2f} {:.2f} : {:.2f} a.u.""".format(
                    point, *massdict[point]["kpoint"], DOSmass
                )
            )

    def get_kcoords_of(self, energy):
        """ This function retrieves the reciprocal coordinates of an energy that matches a band.
        energ : float (self.VBM or self.CBM)
        
        In principle it works for any energy that is part of the spectrum, but it works best for the
        valence band maximum and conduction band minimum. In case of degeneracies, it returns all 
        k-vectors that locate that energy.

        First, the algorithm checks whether the energy matches a high-symmetry point. If not,
        it determines the k-point numerically with an accuracy of 1E-4.
        """
        index = np.where(self.energies == energy)
        nbands = len(
            index[0]
        )  # this checks for multiple extrema with same energies at different k-points
        locs = []
        for i in range(nbands):  # check for every band
            row = index[0][i]
            band = index[1][i]
            kp = self.spectrum[:, 0][row]
            high_symm = False
            for kcoord in range(len(self.klabel_coords)):
                if kp == self.klabel_coords[kcoord]:
                    high_symm = True
                    loc = self.kvectors[self.kpath[kcoord]]
                    locs.append(loc)
            if high_symm == False:
                for k in range(len(self.klabel_coords)):
                    if kp < self.klabel_coords[k]:
                        shift = self.klabel_coords[k - 1]
                        start = self.kvectors[self.kpath[k - 1]] * self.rec_cell_lengths
                        end = self.kvectors[self.kpath[k]] * self.rec_cell_lengths
                        diff = end - start
                        trial = np.arange(0, 1, 0.00005)
                        for value in trial:
                            new = start + (value * diff)
                            norm = np.linalg.norm(new)
                            delta = np.abs(norm - kp - shift)
                            if delta < 0.00005:
                                new = new / (self.rec_cell_lengths)
                                locs.append(np.around(new, 4))
                        break
        uniques = []
        for arr in locs:
            if not any(np.array_equal(arr, unique_arr) for unique_arr in uniques):
                uniques.append(arr)
        return uniques

    def resample(self, locs, label, radius=25):
        """ Constructs band paths sampled around the extreme points stored in self.sample_bands.
        locs : np.array (k-coordinates of the extreme points)
        label : str (e.g. VBM, CBM)
        radius : int (fraction of the reciprocal cell lengths to sample) """
        x, y, z = self.rec_cell_lengths / radius  # cutting off the sampling in x, y, z
        directions = ["xx", "xy", "yy", "xz", "yz", "zz"]
        if list(self.cell.cell[2]) == [0.0, 0.0, 100.0]:
            self.cell.pbc = [True, True, False]
        for loc in locs:
            j = 0
            ### xx case
            start = loc - np.array([x, 0, 0])
            end = loc + np.array([x, 0, 0])
            self.sample_bands[
                ("neg_xx_{}{}".format(label, j), "pos_xx_{}{}".format(label, j))
            ] = (start, end)
            ### xy case
            start = loc - np.array([x, y, 0])
            end = loc + np.array([x, y, 0])
            self.sample_bands[
                ("neg_xy_{}{}".format(label, j), "pos_xy_{}{}".format(label, j))
            ] = (start, end)
            ## yy case
            start = loc - np.array([0, y, 0])
            end = loc + np.array([0, y, 0])
            self.sample_bands[
                ("neg_yy_{}{}".format(label, j), "pos_yy_{}{}".format(label, j))
            ] = (start, end)
            if self.cell.pbc[2] == True:
                ### xz case
                start = loc - np.array([x, 0, z])
                end = loc + np.array([x, 0, z])
                self.sample_bands[
                    ("neg_xz_{}{}".format(label, j), "pos_xz_{}{}".format(label, j))
                ] = (start, end)
                ### yz case
                start = loc - np.array([0, y, z])
                end = loc + np.array([0, y, z])
                self.sample_bands[
                    ("neg_yz_{}{}".format(label, j), "pos_yz_{}{}".format(label, j))
                ] = (start, end)
                ### zz case
                start = loc - np.array([0, 0, z])
                end = loc + np.array([0, 0, z])
                self.sample_bands[
                    ("neg_zz_{}{}".format(label, j), "pos_zz_{}{}".format(label, j))
                ] = (start, end)

    def prepare_new_calc(self):
        """ Creates new directory called eff_mass and writes resampled calculation files
        to this directory. """
        new_dir = os.path.join(str(self.path), "eff_mass")
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        shutil.copy(str(self.path.joinpath("control.in")), new_dir)
        shutil.copy(str(self.path.joinpath("geometry.in")), new_dir)
        shs = list(self.path.glob("*.sh"))
        cscs = list(self.path.glob("*.csc"))
        if shs != []:
            for sh in shs:
                shutil.copy(str(sh), new_dir)
        if cscs != []:
            for csc in cscs:
                shutil.copy(str(csc), new_dir)
        with open(self.path.joinpath("control.in"), "r+") as file:
            control = [line for line in file.readlines() if "output band" not in line]
        with open(self.path.joinpath("eff_mass/control.in"), "w") as file:
            for line in control:
                write = False if line.startswith("#") else True
                if write:
                    if "elsi_restart" in line:
                        line = "elsi_restart    read      100\n"
                    if "k_grid" in line:
                        output_bands = []
                        for i in list(self.sample_bands.keys()):
                            vec1 = "{:6f} {:6f} {:6f}".format(*self.sample_bands[i][0])
                            vec2 = "{:6f} {:6f} {:6f}".format(*self.sample_bands[i][1])
                            output_bands.append(
                                "output band {vec1}    {vec2}  {npoints}  {label1} {label2}".format(
                                    label1=i[0],
                                    label2=i[1],
                                    npoints=31,
                                    vec1=vec1,
                                    vec2=vec2,
                                )
                            )
                        for band in output_bands:
                            line += band + "\n"
                file.write(line)


#################################

# test = eff_mass("AIMS_tools\Tests\MoS2\MoS2.out")
# test2 = eff_mass("AIMS_tools\Tests\MoS2\eff_mass\MoS2.out")

test = eff_mass("AIMS_tools\Tests\Silicon\Si.out")
test = eff_mass("AIMS_tools\Tests\Silicon\eff_mass\Si.out")

=======
import numpy as np
import glob, sys, os, shutil
import matplotlib.pyplot as plt
from pathlib import Path as Path
from scipy import interpolate
from AIMS_tools import bandstructure
import ase.io, ase.cell, ase.spacegroup
import math
from scipy.optimize import curve_fit
import scipy.constants as const


plt.style.use("seaborn-ticks")

### units
hartree_to_eV = 27.211402666173235  # eV to hartree


class eff_mass(bandstructure.bandstructure):
    """ The effective mass calculator inherits from the bandstructure class.
    The calculation has several steps:
    - The k-point coordinates of the VBM and CBM are determined numerically.
    - Sampling paths around these two points are prepared. These consider dimensionality.
    - If needed, the files for a new band structure calculation are prepared. If possible, these
      can be simply restarted as a non-self-consistent calculation, if the keyword
      elsi_restart read_and_write freq was included in the first calculation.
    - The lowest conduction band and highest valence band are fitted with parabolic functions
      for all directions in k-space to evaluate the effective mass components.
    - Quantities derived from the effective mass components are calculated including degeneracy factors.
    """

    def __init__(self, outputfile, get_SOC=True, custom_path="", shift_to="middle"):
        super().__init__(
            outputfile, get_SOC=get_SOC, custom_path=custom_path, shift_to=shift_to
        )
        if custom_path != "":
            self.custom_path(custom_path)

        self.energies = (
            self.spectrum[:, 1:] / hartree_to_eV
        )  # converting to atomic units
        self.sg = ase.spacegroup.get_spacegroup(self.cell)

        ### Execute first steps, if calculation has not been run yet
        if not self.path.joinpath("eff_mass").exists() and "eff_mass" != str(
            self.path.parts[-1]
        ):
            self.calc_dir = self.path.joinpath("eff_mass")
            print(
                "Preparing files for calculation in directory {}.".format(
                    str(self.calc_dir)
                )
            )
            self.VBM = np.max(self.energies[self.energies < 0])
            self.CBM = np.min(self.energies[self.energies > 0])
            self.VBM_locs, self.CBM_locs = (  # list of locations with the same energy
                self.get_kcoords_of(self.VBM),
                self.get_kcoords_of(self.CBM),
            )
            self.sample_bands = {}
            self.resample(self.VBM_locs, "VBM")
            self.resample(self.CBM_locs, "CBM")
            self.prepare_new_calc()
            sys.exit()

        ### Execute if calculation has been run
        if str(self.path.parts[-1]) == "eff_mass":
            print("Evaluating calculations in directory {}.".format(str(self.path)))
            self.masses = self.band_fitting_method(0)
            # self.evaluate_masses(self.masses)
            # if self.active_SOC == True:
            #     self.SOC_masses = self.band_fitting_method(1)

    #                self.evaluate_masses(self.SOC_masses)

    ###   Methods
    def band_fitting_method(self, band_index):
        directions = {}
        ar = np.zeros((int(self.spectrum.shape[0] / 2), 7))
        j = 0
        t = 0
        for axis in list(self.ksections.keys()):
            if "VBM" in axis[0]:  # should be 6 directions
                self.custom_path(axis[0] + "-" + axis[1])
                x = self.spectrum[:, 0]
                y = self.spectrum[:, 1:] / hartree_to_eV
                VBM = np.max(y[y < 0])
                index = np.where(y == VBM)
                loc = index[0][0]
                band = index[1][0]
                band -= band_index
                VBM = y[loc, band]  # this realigns the energy to off-split bands
                end = t + len(x)
                ar[t:end, j] = x
                ar[t:end, 6] = y[:, band]
                j += 1
                t += len(x)

        def tensor(Vars, E0, mxx, myy, mzz, mxy, mxz, myz):
            xx, yy, zz, xy, xz, yz = (
                Vars[:, 0],
                Vars[:, 1],
                Vars[:, 2],
                Vars[:, 3],
                Vars[:, 4],
                Vars[:, 5],
            )
            return (
                E0
                + 1 / (2 * mxx) * xx ** 2
                + 1 / (2 * myy) * yy ** 2
                + 1 / (2 * mzz) * zz ** 2
                + 1 / (2 * mxy) * xy ** 2
                + 1 / (2 * mxz) * xz ** 2
                + 1 / (2 * myz) * yz ** 2
            )

        p0 = VBM, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1
        bounds = (
            (VBM - 0.00001, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf),
            (VBM + 0.00001, -0.0001, -0.0001, -0.0001, -0.0001, -0.0001, -0.0001),
        )
        popt, pcov = curve_fit(tensor, ar[:, 0:6], ar[:, 6], bounds=bounds)
        err = np.sqrt(np.diag(pcov))[1]

        mtens = np.array(
            [
                [popt[1], popt[4], popt[5]],
                [popt[4], popt[2], popt[6]],
                [popt[5], popt[6], popt[3]],
            ]
        )
        # mtens = np.linalg.inv(mtens)
        w, v = np.linalg.eig(mtens)
        print(mtens)
        print(w)
        print(v / self.rec_cell_lengths)
        # _, direction, label = axis[0].split("_")
        # kpoint = tuple((self.kvectors[axis[0]] + self.kvectors[axis[1]]) / 2)
        # degeneracy = len(self.sg.equivalent_lattice_points(kpoint))
        # if label not in masses.keys():
        #     masses[label] = {
        #         "kpoint": kpoint,
        #         direction: popt[1],
        #         "degeneracy": degeneracy,
        #     }
        # else:
        #     masses[label].update(
        #         {"kpoint": kpoint, direction: popt[1], "degeneracy": degeneracy}
        #     )

    def old_band_fitting_method(self, band_index):
        """ Fits parabolic functions for each resampled extreme point
        in x, y and z direction. For each fitting, a plot is saved including
        the fitted mass, the fitting error, and the k-point. The points are labelled as
        VBM or CBM and the band index, where +1 or -1 refers to to the band below or above
        (which is the case for SOC calculations).
        
        band_index : int (1 to obtain VBM-1 and CBM+1)
        """
        masses = {}
        for axis in list(self.ksections.keys()):
            self.custom_path(axis[0] + "-" + axis[1])
            x = self.spectrum[:, 0]
            y = self.spectrum[:, 1:] / hartree_to_eV
            _, direction, label = axis[0].split("_")
            kpoint = tuple((self.kvectors[axis[0]] + self.kvectors[axis[1]]) / 2)
            degeneracy = len(self.sg.equivalent_lattice_points(kpoint))
            if "VBM" in axis[0]:
                VBM = np.max(y[y < 0])
                index = np.where(y == VBM)
                loc = index[0][0]
                band = index[1][0]
                band -= band_index
                VBM = y[loc, band]  # this realigns the energy to off-split bands
                p0 = VBM, -0.1
                bounds = ((VBM - 0.00001, -10), (VBM + 0.00001, 10))
                label = label + "-" + str(band_index)
            if "CBM" in axis[0]:
                CBM = np.min(y[y > 0])
                index = np.where(y == CBM)
                loc = index[0][0]
                band = index[1][0]
                band += band_index
                CBM = y[loc, band]  # this realigns the energy to off-split bands
                p0 = CBM, 0.1
                bounds = ((CBM - 0.00001, -10), (CBM + 0.00001, 10))
                label = label + "+" + str(band_index)
            x = x - x[loc]
            y = y[:, band]

            def parabola(k, E0, m):
                return E0 + ((k ** 2) / (2 * m))

            popt, pcov = curve_fit(parabola, x, y, p0, bounds=bounds)
            err = np.sqrt(np.diag(pcov))[1]

            if label not in masses.keys():
                masses[label] = {
                    "kpoint": kpoint,
                    direction: popt[1],
                    "degeneracy": degeneracy,
                }
            else:
                masses[label].update(
                    {"kpoint": kpoint, direction: popt[1], "degeneracy": degeneracy}
                )
            plt.figure(figsize=(2, 3))
            plt.scatter(x, y, color="crimson", alpha=0.7, s=10)
            plt.plot(x, parabola(x, *popt), color="blue")
            plt.plot(x, self.spectrum[:, 1:] / hartree_to_eV, color="gray", alpha=0.5)
            plt.xlim([np.min(x), np.max(x)])
            plt.ylim([np.min(y) - 0.001, np.max(y) + 0.001])
            plt.xlabel("$\Delta$k$_" + direction + "$ [2$\pi$/bohr]")
            plt.ylabel("E-E$_F$ [Hartree]")
            if "VBM" in label:
                plt.title("VBM -" + str(band_index))
                plt.annotate(
                    """k = ({:.2f} {:.2f} {:.2f}) \nm = {:.2f}$\pm${:.2f}""".format(
                        *kpoint, popt[1], err
                    ),
                    xy=(np.min(x) + 0.015, np.min(y) - 0.0005),
                )
                os.chdir(str(self.path))
                plt.savefig(
                    "VBM_-" + str(band_index) + direction + ".png",
                    dpi=300,
                    bbox_inches="tight",
                    faceolor="white",
                    transparent=False,
                )
            if "CBM" in label:
                plt.title("CBM +" + str(band_index))
                plt.annotate(
                    """k = ({:.2f} {:.2f} {:.2f})\nm = {:.2f}$\pm${:.2f}""".format(
                        *kpoint, popt[1], err
                    ),
                    xy=(np.min(x) + 0.015, np.max(y)),
                )
                os.chdir(str(self.path))
                plt.savefig(
                    "CBM_+" + str(band_index) + direction + ".png",
                    dpi=300,
                    bbox_inches="tight",
                    faceolor="white",
                    transparent=False,
                )
        return masses

    def evaluate_masses(self, massdict):
        print("Calculating mass-component dervied quantities.")
        for point in massdict.keys():
            masses = []
            directions = [
                i for i in list(massdict[point].keys()) if i in ["x", "y", "z"]
            ]
            for i in directions:
                masses.append(np.abs(massdict[point][i]))
            DOSmass = (massdict[point]["degeneracy"] ** 2 * np.prod(masses)) ** (
                1 / len(masses)
            )
            print(
                """DOS mass of {} at K-Point {:.2f} {:.2f} {:.2f} : {:.2f} a.u.""".format(
                    point, *massdict[point]["kpoint"], DOSmass
                )
            )

    def get_kcoords_of(self, energy):
        """ This function retrieves the reciprocal coordinates of an energy that matches a band.
        energ : float (self.VBM or self.CBM)
        
        In principle it works for any energy that is part of the spectrum, but it works best for the
        valence band maximum and conduction band minimum. In case of degeneracies, it returns all 
        k-vectors that locate that energy.

        First, the algorithm checks whether the energy matches a high-symmetry point. If not,
        it determines the k-point numerically with an accuracy of 1E-4.
        """
        index = np.where(self.energies == energy)
        nbands = len(
            index[0]
        )  # this checks for multiple extrema with same energies at different k-points
        locs = []
        for i in range(nbands):  # check for every band
            row = index[0][i]
            band = index[1][i]
            kp = self.spectrum[:, 0][row]
            high_symm = False
            for kcoord in range(len(self.klabel_coords)):
                if kp == self.klabel_coords[kcoord]:
                    high_symm = True
                    loc = self.kvectors[self.kpath[kcoord]]
                    locs.append(loc)
            if high_symm == False:
                for k in range(len(self.klabel_coords)):
                    if kp < self.klabel_coords[k]:
                        shift = self.klabel_coords[k - 1]
                        start = self.kvectors[self.kpath[k - 1]] * self.rec_cell_lengths
                        end = self.kvectors[self.kpath[k]] * self.rec_cell_lengths
                        diff = end - start
                        trial = np.arange(0, 1, 0.00005)
                        for value in trial:
                            new = start + (value * diff)
                            norm = np.linalg.norm(new)
                            delta = np.abs(norm - kp - shift)
                            if delta < 0.00005:
                                new = new / (self.rec_cell_lengths)
                                locs.append(np.around(new, 4))
                        break
        uniques = []
        for arr in locs:
            if not any(np.array_equal(arr, unique_arr) for unique_arr in uniques):
                uniques.append(arr)
        return uniques

    def resample(self, locs, label, radius=25):
        """ Constructs band paths sampled around the extreme points stored in self.sample_bands.
        locs : np.array (k-coordinates of the extreme points)
        label : str (e.g. VBM, CBM)
        radius : int (fraction of the reciprocal cell lengths to sample) """
        x, y, z = self.rec_cell_lengths / radius  # cutting off the sampling in x, y, z
        directions = ["xx", "xy", "yy", "xz", "yz", "zz"]
        if list(self.cell.cell[2]) == [0.0, 0.0, 100.0]:
            self.cell.pbc = [True, True, False]
        for loc in locs:
            j = 0
            ### xx case
            start = loc - np.array([x, 0, 0])
            end = loc + np.array([x, 0, 0])
            self.sample_bands[
                ("neg_xx_{}{}".format(label, j), "pos_xx_{}{}".format(label, j))
            ] = (start, end)
            ### xy case
            start = loc - np.array([x, y, 0])
            end = loc + np.array([x, y, 0])
            self.sample_bands[
                ("neg_xy_{}{}".format(label, j), "pos_xy_{}{}".format(label, j))
            ] = (start, end)
            ## yy case
            start = loc - np.array([0, y, 0])
            end = loc + np.array([0, y, 0])
            self.sample_bands[
                ("neg_yy_{}{}".format(label, j), "pos_yy_{}{}".format(label, j))
            ] = (start, end)
            if self.cell.pbc[2] == True:
                ### xz case
                start = loc - np.array([x, 0, z])
                end = loc + np.array([x, 0, z])
                self.sample_bands[
                    ("neg_xz_{}{}".format(label, j), "pos_xz_{}{}".format(label, j))
                ] = (start, end)
                ### yz case
                start = loc - np.array([0, y, z])
                end = loc + np.array([0, y, z])
                self.sample_bands[
                    ("neg_yz_{}{}".format(label, j), "pos_yz_{}{}".format(label, j))
                ] = (start, end)
                ### zz case
                start = loc - np.array([0, 0, z])
                end = loc + np.array([0, 0, z])
                self.sample_bands[
                    ("neg_zz_{}{}".format(label, j), "pos_zz_{}{}".format(label, j))
                ] = (start, end)

    def prepare_new_calc(self):
        """ Creates new directory called eff_mass and writes resampled calculation files
        to this directory. """
        new_dir = os.path.join(str(self.path), "eff_mass")
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        shutil.copy(str(self.path.joinpath("control.in")), new_dir)
        shutil.copy(str(self.path.joinpath("geometry.in")), new_dir)
        shs = list(self.path.glob("*.sh"))
        cscs = list(self.path.glob("*.csc"))
        if shs != []:
            for sh in shs:
                shutil.copy(str(sh), new_dir)
        if cscs != []:
            for csc in cscs:
                shutil.copy(str(csc), new_dir)
        with open(self.path.joinpath("control.in"), "r+") as file:
            control = [line for line in file.readlines() if "output band" not in line]
        with open(self.path.joinpath("eff_mass/control.in"), "w") as file:
            for line in control:
                write = False if line.startswith("#") else True
                if write:
                    if "elsi_restart" in line:
                        line = "elsi_restart    read      100\n"
                    if "k_grid" in line:
                        output_bands = []
                        for i in list(self.sample_bands.keys()):
                            vec1 = "{:6f} {:6f} {:6f}".format(*self.sample_bands[i][0])
                            vec2 = "{:6f} {:6f} {:6f}".format(*self.sample_bands[i][1])
                            output_bands.append(
                                "output band {vec1}    {vec2}  {npoints}  {label1} {label2}".format(
                                    label1=i[0],
                                    label2=i[1],
                                    npoints=31,
                                    vec1=vec1,
                                    vec2=vec2,
                                )
                            )
                        for band in output_bands:
                            line += band + "\n"
                file.write(line)


#################################

# test = eff_mass("AIMS_tools\Tests\MoS2\MoS2.out")
# test2 = eff_mass("AIMS_tools\Tests\MoS2\eff_mass\MoS2.out")

test = eff_mass("AIMS_tools\Tests\Silicon\Si.out")
test = eff_mass("AIMS_tools\Tests\Silicon\eff_mass\Si.out")

>>>>>>> a81894f55b2a444765ab6824b492a47998392be1
