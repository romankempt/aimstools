import ase.io, ase.cell, ase.spacegroup
import math
import scipy
from scipy.optimize import curve_fit, leastsq, least_squares
from scipy.interpolate import NearestNDInterpolator as ndi
from scipy.interpolate import LinearNDInterpolator as ldi
from scipy import spatial
from scipy import interpolate, misc

from AIMS_tools.misc import *
from AIMS_tools.postprocessing import postprocess


class eff_mass(postprocess):
    """Class to evaluate effective masses.

    Employs band fitting algorithm combined with finite differences to calculate effective masses in 2 or 3 dimensions.
    
    Args:
        VBM (bool): Evaluate effective masses for valence band maximum. Defaults to true.
        CBM (bool): Evaluate effective masses for conduction minimum. Defaults to true.
        scale (int): Number of steps to take in each direction around evaluated point.
        nbands (int): Number of bands to evaluate below or above the extreme point.
    
    Note:
        3D boundary conditions are problematic.
    """

    def __init__(
        self, outputfile, get_SOC=True, spin=None, VBM=True, CBM=True, scale=3, nbands=1
    ):
        super().__init__(outputfile, get_SOC=get_SOC, spin=spin)
        if self.spin == None:
            spin = 0
        logging.info("Reading in kpoints and eigenvalues ...")
        self.kpoints, self.eigenvalues = self.__read_AIMS_eigenvalues(
            self.path.joinpath("Final_KS_eigenvalues.dat")
        )
        is_2D = self.structure.is_2d(self.structure.atoms)
        self.pbc = "3D" if is_2D == False else "2D"
        self.scale = scale
        self.__determine_stepsize(scale)

        if VBM == True:
            self.__VBM_routine(spin, nbands=nbands)
        if CBM == True:
            self.__CBM_routine(spin, nbands=nbands)

    def __VBM_routine(self, spin, nbands=1):
        """ Performs effective mass evaluation for the valence band maximum.

        Automatically detects and analysis VBM.

        Args:
            spin (int): 0 or 1 for spin channel. Defaults to 0.
            nbands (int): Number of bands to perform analysis, couting down from the VBM band. Defaults to 1.

        Attributes:
            VBM (float): VBM energy in Hartree
            VBMloc (ndarray): Fractional k-point coordinates of VBM.
            VBMband (int): Band index of VBM.
            cut (ndarray): Cutted band range around VBMloc.
        """
        logging.info("Initializing routine for valence bands ...")
        self.VBM = np.max(
            self.eigenvalues[spin, :, :][self.eigenvalues[spin, :, :] < 0]
        )
        self.VBMloc, self.VBMband = self.__find_extreme(self.VBM, spin=spin)
        self.check_energy_equivalence(self.VBMloc, self.VBM, spin=spin)
        for j in range(nbands):
            logging.info("---- VBM routine ----")
            logging.info(
                "Beginning effective mass evaluation for band {}:".format(
                    self.VBMband - j
                )
            )
            self.cut = self.__cut_box(self.VBMloc, self.VBMband - j, scale=self.scale)
            if self.pbc == "2D":
                fit, g = self.__fit_2D(self.cut, self.VBMloc)
                self.axplot_2D(fit, g, self.cut)
            elif self.pbc == "3D":
                fit, g = self.__fit_3D(self.cut, self.VBMloc)
                self.axplot_3D(fit, g, self.cut)
        logging.info("----------------")

    def __CBM_routine(self, spin, nbands=1):
        """ Performs effective mass evaluation for the conduction band minimum.

        Automatically detects and analysis CBM.

        Args:
            spin (int): 0 or 1 for spin channel. Defaults to 0.
            nbands (int): Number of bands to perform analysis, couting up from the CBM band. Defaults to 1.

        Attributes:
            CBM (float): CBM energy in Hartree
            CBMloc (ndarray): Fractional k-point coordinates of CBM.
            CBMband (int): Band index of CBM.
            cut (ndarray): Cutted band range around CBMloc.
        """
        logging.info("Initializing routine for conduction bands ...")
        self.CBM = np.min(
            self.eigenvalues[spin, :, :][self.eigenvalues[spin, :, :] > 0]
        )
        self.CBMloc, self.CBMband = self.__find_extreme(self.CBM, spin=spin)
        self.check_energy_equivalence(self.CBMloc, self.CBM, spin=spin)
        for j in range(nbands):
            logging.info("---- CBM routine ----")
            logging.info(
                "Beginning effective mass evaluation for band {}:".format(
                    self.CBMband + j
                )
            )
            self.cut = self.__cut_box(self.CBMloc, self.CBMband + j, scale=self.scale)
            if self.pbc == "2D":
                fit, g = self.__fit_2D(self.cut, self.CBMloc)
                self.axplot_2D(fit, g, self.cut)
            elif self.pbc == "3D":
                fit, g = self.__fit_3D(self.cut, self.CBMloc)
                self.axplot_3D(fit, g, self.cut)
        logging.info("----------------")

    def point_routine(self, point, energy, band, spin, nbands=1):
        """ Performs effective mass evaluation for the user-defined point.

        Algorithm only yields senseful results for parabolic extrema.

        Example:
            >>> from AIMS_tools import eff_mass
            >>> efm = AIMS_tools.eff_mass(directory)
            >>> point, VBM, VBM_band, CBM, CBM_band = efm.pick_band([1/3,1/3,0])
            >>> efm.point_routine(point, VBM, VBM_band, spin=0, nbands=1)
            >>> efm.point_routine(point, CBM, CBM_band, spin=0, nbands=1)

        Args:
            energy (float): Energy of extreme point.
            band (int): Band index for extreme point.
            spin (int): 0 or 1 for spin channel. Defaults to 0.
            nbands (int): Number of bands to perform analysis, couting up from the CBM band. Defaults to 1.

        """
        logging.info(
            "Initializing routine for user-defined point k = ( {: 10.6f} {: 10.6f} {: 10.6f} ) :".format(
                *point
            )
        )
        self.check_energy_equivalence(point, energy, spin=spin)
        for j in range(nbands):
            j = j if energy < 0 else (-1) * j
            logging.info("---- point routine ----")
            logging.info(
                "Beginning effective mass evaluation for band {}:".format(band + j)
            )
            self.cut = self.__cut_box(point, band + j, scale=self.scale)
            if self.pbc == "2D":
                fit, g = self.__fit_2D(self.cut, point)
                self.axplot_2D(fit, g, self.cut)
            elif self.pbc == "3D":
                fit, g = self.__fit_3D(self.cut, point)
                self.axplot_3D(fit, g, self.cut)
        logging.info("----------------")

    def __read_AIMS_eigenvalues(self, filename="Final_KS_eigenvalues.dat"):
        """Read in a Final_KS_eigenvalues.dat file in AIMS format.

        The file contains blocks of k-points with eigenvalues and occupations.
        FHI-AIMS writes out the eigenvalues for the entire first Billouin Zone.
        By default, it runs on a Gamma-centered grid.
        If SOC is enabled, an additional file called Final_KS_eigenvalues.dat.no_soc is present
        containing unperturbed eigenvalues.
        If SOC is enabled, every eigenvalue is split and singly occupied.
        If collinear spin is enabled, there is an additional block for spin up / spin down.

        Args:
            filename (str): path to the Final_KS_eigenvalues.dat file

        Returns:
            tuple : An (nkpoints, 3) array with the coordinates with the k points in the file and an (nspins, nkpoints, nbands) array containing band energies in Ha.
        """

        with open(filename, "r") as file:
            content = file.readlines()

        logging.info("Centering all k-points into 1st Brillouin Zone...")
        # kpoints
        kpoints = [
            line.split()[-3:]
            for line in content
            if "k-point in recip. lattice units:" in line
        ]
        kpoints = np.array(kpoints, dtype=float)
        for row in range(kpoints.shape[0]):
            if kpoints[row][0] > 0.5:
                kpoints[row][0] -= 1
            if kpoints[row][1] > 0.5:
                kpoints[row][1] -= 1
            if kpoints[row][2] > 0.5:
                kpoints[row][2] -= 1

        # eigenvalues
        from itertools import groupby

        logging.info("Grouping all eigenvalues per k-point...")
        content = [
            line.strip().split()
            for line in content
            if "k-point" not in line
            and "#" not in line
            and "occupation number (dn), eigenvalue (dn)" not in line
        ]
        content = [
            list(group) for k, group in groupby(content, lambda x: x == []) if not k
        ]  # this splits by empty lines
        if (self.active_SOC == False) and (self.spin != None):
            spinup = (
                np.array(content, dtype=float)[:, :, 2] * hartree
            )  # (nkpoints, bands)
            spindown = (
                np.array(content, dtype=float)[:, :, 4] * hartree
            )  # (nkpoints, bands)
        else:
            content = (
                np.array(content, dtype=float)[:, :, 2] * hartree
            )  # (nkpoints, bands)

        if (self.active_SOC == False) and (self.spin != None):
            nbands = spinup.shape[1]
            nks = spinup.shape[0]
            nbands2 = spindown.shape[1]
            assert (
                nbands == nbands2
            ), "Number of spin-up and spin-down bands is not the same."
        else:
            nbands = int(content.shape[1])
            nks = content.shape[0]

        ebands = np.empty((2, nks, nbands))

        if (self.active_SOC == False) and (self.spin != None):
            ebands[0, :, :] = spinup - (self.fermi_level * hartree)
            ebands[1, :, :] = spindown - (self.fermi_level * hartree)
        else:
            ebands[0, :, :] = content - (self.fermi_level * hartree)

        return kpoints, ebands

    def check_energy_equivalence(self, point, energy, spin=0):
        """ Checks energy equivalencies of given point. """

        def closest_point(point, points):
            tree = spatial.KDTree(points)
            idx = tree.query(point)[1]
            return idx

        sg = ase.spacegroup.get_spacegroup(self.structure.atoms, symprec=1e-4)
        sites = sg.equivalent_lattice_points(point)
        for row in range(sites.shape[0]):
            if sites[row][0] > 0.5:
                sites[row][0] -= 1
            if sites[row][1] > 0.5:
                sites[row][1] -= 1
            if sites[row][2] > 0.5:
                sites[row][2] -= 1
            if sites[row][0] <= -0.5:
                sites[row][0] += 1
            if sites[row][1] <= -0.5:
                sites[row][1] += 1
            if sites[row][2] <= -0.5:
                sites[row][2] += 1
        mpoint = -1 * point
        logging.info(
            "Point k = ( {: .6f} {: .6f} {: .6f} ) is {} times degenerate.".format(
                *point, len(sites)
            )
        )
        for site in sites:
            if not np.allclose(point, site, rtol=1e-4, atol=1e-6) and not np.allclose(
                mpoint, site, rtol=1e-4, atol=1e-6
            ):
                idx = closest_point(site, self.kpoints)
                if energy < 0:
                    ev_at_site = np.max(
                        self.eigenvalues[spin, idx, :][
                            self.eigenvalues[spin, idx, :] < 0
                        ]
                    )
                elif energy > 0:
                    ev_at_site = np.min(
                        self.eigenvalues[spin, idx, :][
                            self.eigenvalues[spin, idx, :] > 0
                        ]
                    )
                delta = energy - ev_at_site
                dloc = self.kpoints[idx]
                if delta > 1e-4 or delta < 1e-4:
                    logging.warning(
                        "\t Energy at symmetry-equivalent point k = ( {: .6f} {: .6f} {: .6f} ) differs from energy at closest point on grid k' = ( {: .6f} {: .6f} {: .6f} ) by {: .4f} Hartree.".format(
                            *site, *dloc, delta
                        )
                    )
        else:
            logging.info(
                "Grid finished all symmetry equivalency checks. Warnings can be ignored for now."
            )

    def __find_extreme(self, energy, spin=0):
        """ Finds extreme point for given energy and spin. 
        
        Performs additional symmetry analyses for degeneracies.

        Args:
            energy (float): Energy value to find.
            spin (int): Spin channel.

        Returns:
            tuple : location as ndarray and band index as integer.
        """
        sg = ase.spacegroup.get_spacegroup(self.structure.atoms, symprec=1e-4)

        index = np.argwhere(
            np.isclose(self.eigenvalues[spin, :, :], energy, rtol=1e-4, atol=1e-6)
        )
        if index.shape[0] > 1:
            locs = [self.kpoints[index[0, 0]]]
            sites, _ = sg.equivalent_sites(self.kpoints[index[0, 0]])
            for point in range(1, index.shape[0]):
                loc = self.kpoints[index[point, 0]]
                if loc.tolist() not in sites:
                    locs.append(loc)
            if len(locs) > 1:
                logging.warning(
                    "Found {} symmetry non-equivalent extrema.".format(len(locs))
                )
                loc, band = locs[0], index[0, 1]
                logging.warning(
                    "\t \t Considering only one of them: \t k = ( {: .6f} {: .6f} {: .6f} )".format(
                        point, *loc
                    )
                )
            else:
                loc, band = locs[0], index[0, 1]
        else:
            loc, band = self.kpoints[index[0, 0]], index[0, 1]

        logging.info(
            "Found band extremum at k = ( {: .6f} {: .6f} {: .6f} ) for band Nr. {} with energy {: .6f} Hartree.".format(
                *loc, band, energy
            )
        )

        return loc, band

    def __determine_stepsize(self, scale=3.0):
        """ Determins stepsize based on minimum distance between points on k-grid and scale factor."""
        self.xstep = min(
            [
                np.abs(x - self.kpoints[0, 0])
                for x in self.kpoints[:, 0]
                if np.abs(x - self.kpoints[0, 0]) >= 1e-5
            ]
        )
        self.ystep = min(
            [
                np.abs(x - self.kpoints[0, 1])
                for x in self.kpoints[:, 1]
                if np.abs(x - self.kpoints[0, 1]) >= 1e-5
            ]
        )
        if self.pbc == "3D":
            self.zstep = min(
                [
                    np.abs(x - self.kpoints[0, 2])
                    for x in self.kpoints[:, 2]
                    if np.abs(x - self.kpoints[0, 2]) >= 1e-5
                ]
            )
        else:
            self.zstep = 0.00000

        logging.info(
            "Step sizes: \n \t \t x-axis: {: 10.6f} [a.u.] \n \t \t y-axis: {: 10.6f} [a.u.] \n \t \t z-axis: {: 10.6f} [a.u.]".format(
                self.xstep, self.ystep, self.zstep
            )
        )
        if self.pbc == "2D":
            area = self.xstep * self.ystep * scale ** 2
            logging.info(
                "Reciprocal area for analysis is {: .6f} [1 / bohr^2] large.".format(
                    area
                )
            )
        elif self.pbc == "3D":
            vol = self.xstep * self.ystep * self.zstep * scale ** 3
            logging.info(
                "Reciprocal volume for analysis is {: .6f} [1 / bohr^3] large.".format(
                    vol
                )
            )

    def __cut_box(self, loc, band, scale=3.1, spin=0):
        """ Cuts 2D or 3D area / volume around point. 
        
        Returns:
            ndarray : (nkpoints, ndims + energy)
        """

        kpoints = self.kpoints - loc
        indices = np.argwhere(
            (
                (np.abs(kpoints[:, 0]) <= (scale * self.xstep) * 1.05)
                | (np.abs(kpoints[:, 0] - 1) <= (scale * self.xstep) * 1.05)
            )
            & (
                (np.abs(kpoints[:, 1]) <= (scale * self.ystep) * 1.05)
                | (np.abs(kpoints[:, 1] - 1) <= (scale * self.ystep) * 1.05)
            )
            & (
                (np.abs(kpoints[:, 2]) <= (scale * self.zstep) * 1.05)
                | (np.abs(kpoints[:, 2] - 1) <= (scale * self.zstep) * 1.05)
            )
        )
        kpoints = kpoints[indices[:, 0]]
        for row in range(kpoints.shape[0]):
            if kpoints[row][0] > 0.5:
                kpoints[row][0] -= 1
            if kpoints[row][1] > 0.5:
                kpoints[row][1] -= 1
            if kpoints[row][2] > 0.5:
                kpoints[row][2] -= 1
            if kpoints[row][0] <= -0.5:
                kpoints[row][0] += 1
            if kpoints[row][1] <= -0.5:
                kpoints[row][1] += 1
            if kpoints[row][2] <= -0.5:
                kpoints[row][2] += 1
        kpoints *= 2 * np.pi / (self.structure.atoms.cell.lengths() * bohr)
        eigenvalues = self.eigenvalues[spin, indices[:, 0], band]
        ar = np.column_stack((kpoints, eigenvalues))
        if self.pbc == "2D":
            ar = ar[ar[:, 2] == 0]
            ar = ar[:, [0, 1, 3]]
        return ar

    def __gridvalue_2D(self, x, y):
        vec = np.array([x * self.xstep, y * self.ystep])
        return scipy.interpolate.griddata(
            self.cut[:, [0, 1]], self.cut[:, 2], vec, method="linear"
        )[0]

    def __fit_2D(self, cut, loc):
        f = self.__gridvalue_2D
        k = 1 / (self.xstep * self.ystep * 4)
        mxx0 = k * (f(2, 0) - 2 * f(0, 0) + f(-2, 0))
        mxy0 = k * (f(1, 1) - f(1, -1) - f(-1, 1) + f(-1, -1))
        myy0 = k * (f(0, 2) - 2 * f(0, 0) + f(0, -2))
        logging.info(
            "Finite Difference estimate: \n \t d^2/dx^2: {: 10.6f} \n \t d^2/dxdy: {: 10.6f} \n \t d^2/dy^2: {: 10.6f}".format(
                mxx0, mxy0, myy0
            )
        )
        try:
            mtens = np.array([[mxx0, mxy0], [mxy0, myy0]])
            mtens = np.linalg.inv(mtens)
            w, v = np.linalg.eig(mtens)
            logging.info(
                "Finite Difference eigenvalues: \n \t {: 10.4f} [m_e] \n \t {: 10.4f} [m_e]".format(
                    *w
                )
            )
        except np.linalg.LinAlgError:
            mxx0, mxy0, myy0 = 100, 100, 100

        def g(x, y, mxxI, mxyI, myyI, mx, my):
            out = mxxI * x * x
            out += 2 * mxyI * x * y
            out += myyI * y * y
            out += mx * x + my * y
            out += f(0, 0)
            return out

        def residuals(params, indata):
            out = list()
            for x, y, v in indata:
                weight = 1 / (1 + np.sqrt(x ** 2 + y ** 2))
                out.append((v - g(x, y, *params)) * weight)
            return out

        guess = [mxx0, mxy0, myy0, 0, 0]
        sol, cov, info, msg, ier = leastsq(
            residuals,
            guess,
            args=(cut,),
            full_output=True,
            factor=50,
            ftol=1e-8,
            xtol=1e-8,
        )
        mxx, mxy, myy, mx, my = [x for x in sol]
        mtens = np.array([[2 * mxx, mxy], [mxy, 2 * myy]])
        mtens = np.linalg.inv(mtens)
        logging.info(
            "Fit derivatives: \n \t d^2/dx^2: {: 10.6f} \n \t d^2/dxdy: {: 10.6f} \n \t d^2/dy^2: {: 10.6f}".format(
                2 * mxx, mxy, 2 * myy
            )
        )
        w, vl = scipy.linalg.eigh(mtens, eigvals_only=False)
        vl = (
            vl * (self.structure.atoms.cell.lengths()[[0, 1]] * 1 / bohr) / (2 * np.pi)
            + loc[[0, 1]]
        )
        logging.info(
            """Fit eigenvalues and eigenvectors: 
            e1 = {: 10.4f} [m_e] \t v1 = ( {: 10.6f} {: 10.6f} )
            e2 = {: 10.4f} [m_e] \t v2 = ( {: 10.6f} {: 10.6f} )""".format(
                w[0], *vl[0], w[1], *vl[1]
            )
        )
        return (sol, g)

    def axplot_2D(self, fitparameters, fitfunction, data):
        """ Plots fit quality along main axes and diagonals in 2D."""
        masses = [x for x in fitparameters]
        g = fitfunction
        fig = plt.figure()
        ax = dict()
        for i in range(1, 5):
            ax[i] = fig.add_subplot(2, 2, i)

        dl = np.linspace(-self.scale, self.scale, 25)
        data = self.cut
        #### xx
        xdata = [[x, v] for x, y, v in data if (abs(y) < 1e-5)]
        vl = np.fromiter((g(x, 0, *masses) for x in dl * self.xstep), np.float)
        ax[1].plot(*zip(*sorted(xdata)), ls="", marker="o", label="data")
        ax[1].plot(dl * self.xstep, vl, color="crimson", label="fit")
        ax[1].set_ylabel(r"E - E$_F$ [a. u.]")
        ax[1].set_xlabel(r"$\Delta k_{xx}$ [a. u.]")
        ax[1].set_title("xx direction")

        #### xy
        xydata = [[x, v] for x, y, v in data if (abs(x - y) < 1e-5)]
        vl = np.fromiter((g(xy, xy, *masses) for xy in dl * self.xstep), np.float)
        ax[2].plot(*zip(*sorted(xydata)), ls="", marker="o", label="data")
        ax[2].plot(dl * self.xstep, vl, color="crimson", label="fit")
        ax[3].plot(*zip(*sorted(xydata)), ls="", marker="o", label="data")
        ax[3].plot(dl * self.ystep, vl, color="crimson", label="fit")
        ax[2].set_ylabel(r"E - E$_F$ [a. u.]")
        ax[2].set_xlabel(r"$\Delta k_{xy}$ [a. u.]")
        ax[2].set_title("xy direction")
        ax[3].set_ylabel(r"E - E$_F$ [a. u.]")
        ax[3].set_xlabel(r"$\Delta k_{yx}$ [a. u.]")
        ax[3].set_title("yx direction")

        #### yy
        yydata = [[y, v] for x, y, v in data if (abs(x) < 1e-5)]
        vl = np.fromiter((g(0, y, *masses) for y in dl * self.ystep), np.float)
        ax[4].plot(*zip(*sorted(yydata)), ls="", marker="o", label="data")
        ax[4].plot(dl * self.ystep, vl, color="crimson", label="fit")
        ax[4].set_ylabel(r"E - E$_F$ [a. u.]")
        ax[4].set_xlabel(r"$\Delta k_{yy}$ [a. u.]")
        ax[4].set_title("yy direction")
        ax[4].legend()

        plt.tight_layout()
        plt.show()

    def planeplot_2D(self, fitparameters, fitfunction, cut):
        """ Plots dispersion relation as two-dimensional planes for xy."""
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm

        masses = [x for x in fitparameters]
        g = fitfunction
        fig = plt.figure()
        ax = Axes3D(fig)
        x = cut[:, 0]
        y = cut[:, 1]
        z = cut[:, 2]
        zf = g(x, y, *masses) - z + np.min(z)
        surf = ax.plot_trisurf(x, y, z, cmap=cm.viridis, linewidth=0.1)
        fit = ax.plot_trisurf(x, y, zf, cmap=cm.plasma, linewidth=0.1)
        fig.colorbar(fit, shrink=0.5, aspect=5)
        ax.set_title("2D fit (difference below)")
        ax.set_xlabel(r"$\Delta k_{xx}$ [a.u.]")
        ax.set_ylabel(r"$\Delta k_{yy}$ [a.u.]")
        ax.set_zlabel(r"E - E$_F$ [a.u.]")
        plt.show()

    def __gridvalue_3D(self, x, y, z):
        vec = np.array([x * self.xstep, y * self.ystep, z * self.ystep])
        return scipy.interpolate.griddata(
            self.cut[:, [0, 1, 2]], self.cut[:, 3], vec, method="linear"
        )[0]

    def __fit_3D(self, cut, loc):
        f = self.__gridvalue_3D

        mxx0 = (
            1
            / (self.xstep * self.xstep * 4)
            * (f(2, 0, 0) - 2 * f(0, 0, 0) + f(-2, 0, 0))
        )
        mxy0 = (
            1
            / (self.xstep * self.ystep * 4)
            * (f(1, 1, 0) - f(1, -1, 0) - f(-1, 1, 0) + f(-1, -1, 0))
        )
        mxz0 = (
            1
            / (self.xstep * self.zstep * 4)
            * (f(1, 0, 1) - f(1, 0, -1) - f(-1, 0, 1) + f(-1, 0, -1))
        )
        myy0 = (
            1
            / (self.ystep * self.ystep * 4)
            * (f(0, 2, 0) - 2 * f(0, 0, 0) + f(0, -2, 0))
        )
        myz0 = (
            1
            / (self.ystep * self.zstep * 4)
            * (f(0, 1, 1) - f(0, 1, -1) - f(0, -1, 1) + f(0, -1, -1))
        )
        mzz0 = (
            1
            / (self.zstep * self.zstep * 4)
            * (f(0, 0, 2) - 2 * f(0, 0, 0) + f(0, 0, -2))
        )

        logging.info(
            """Finite Difference estimate:
            d^2/dx^2:   {: 10.6f} \t d^2/dxdy:   {: 10.6f} \t d^2/dxdz:   {: 10.6f}
            d^2/dxdy:   {: 10.6f} \t d^2/dy^2:   {: 10.6f} \t d^2/dydz:   {: 10.6f}
            d^2/dxdz:   {: 10.6f} \t d^2/dydz:   {: 10.6f} \t d^2/dzdz:   {: 10.6f}""".format(
                mxx0, mxy0, mxz0, mxy0, myy0, myz0, mxz0, myz0, mzz0
            )
        )
        mtens = np.array([[mxx0, mxy0, mxz0], [mxy0, myy0, myz0], [mxz0, myz0, mzz0]])
        mtens = np.linalg.inv(mtens)
        w, v = np.linalg.eig(mtens)
        logging.info(
            "Finite Difference eigenvalues: \n \t {: .4f} [m_e] \n \t {: .4f} [m_e] \n \t {: .4f} [m_e]".format(
                *w
            )
        )

        def g(x, y, z, mxxI, mxyI, mxzI, myyI, myzI, mzzI, mx, my, mz):
            out = mxxI * x * x
            out += 2 * mxyI * x * y
            out += 2 * mxzI * x * z
            out += myyI * y * y
            out += 2 * myzI * y * z
            out += mzzI * z * z
            out += mx * x + my * y + mz * z
            out += f(0, 0, 0)
            return out

        def residuals(params, indata):
            out = list()
            for x, y, z, v in indata:
                val = v - g(x, y, z, *params)
                weight = 1 / (1 + 0.5 * np.sqrt(x ** 2 + y ** 2 + z ** 2))
                out.append(val * weight)
            return out

        guess = [mxx0, mxy0, mxz0, myy0, myz0, mzz0, 0, 0, 0]  # check if inverse or not
        sol, cov, info, msg, ier = leastsq(
            residuals,
            guess,
            args=(cut,),
            full_output=True,
            factor=25,
            ftol=1e-8,
            xtol=1e-8,
        )

        mxx, mxy, mxz, myy, myz, mzz, mx, my, mz = [x for x in sol]
        mtens = np.array(
            [[2 * mxx, mxy, mxz], [mxy, 2 * myy, myz], [mxz, myz, 2 * mzz]]
        )
        mtens = np.linalg.inv(mtens)
        logging.info(
            """Fit derivatives:
            d^2/dx^2: {: 10.6f} \t d^2/dxdy: {: 10.6f} \t d^2/dxdz: {: 10.6f}
            d^2/dxdy: {: 10.6f} \t d^2/dy^2: {: 10.6f} \t d^2/dydz: {: 10.6f}
            d^2/dxdz: {: 10.6f} \t d^2/dydz: {: 10.6f} \t d^2/dzdz: {: 10.6f}""".format(
                2 * mxx, mxy, mxz, mxy, 2 * myy, myz, mxz, myz, 2 * mzz
            )
        )
        w, vl = scipy.linalg.eigh(mtens, eigvals_only=False)
        vl = vl * (self.structure.atoms.cell.lengths() * 1 / bohr) / (2 * np.pi) + loc
        logging.info(
            """Fit eigenvalues and eigenvectors: 
            e1 = {: 10.4f} [m_e] \t v1 = ( {: 10.6f} {: 10.6f} )
            e2 = {: 10.4f} [m_e] \t v2 = ( {: 10.6f} {: 10.6f} )
            e3 = {: 10.4f} [m_e] \t v3 = ( {: 10.6f} {: 10.6f} )""".format(
                w[0], *vl[0], w[1], *vl[1], w[2], *vl[2]
            )
        )
        return (sol, g)

    def plot_isosurface_3d(self, fitparameters):
        """ Plots dispersion relation as 3D isosurface.
        
        Note:
            Requires sci-kit image library.
         """
        energy = self.__gridvalue_3D(0, 0, 0)

        def g(x, y, z, mxxI, mxyI, mxzI, myyI, myzI, mzzI, mx, my, mz):
            out = mxxI * x * x
            out += 2 * mxyI * x * y
            out += 2 * mxzI * x * z
            out += myyI * y * y
            out += 2 * myzI * y * z
            out += mzzI * z * z
            out += mx * x + my * y + mz * z
            out += energy
            return out

        from skimage import measure
        from mpl_toolkits.mplot3d import Axes3D

        a = 4 * self.xstep
        b = 4 * self.ystep
        c = 4 * self.zstep
        x, y, z = np.mgrid[-a:a:31j, -b:b:31j, -c:c:31j]
        vol = g(x, y, z, *fitparameters)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for ival in [0.4, 0.45, 0.5, 0.55]:
            isovalue = ival * ((vol.max() + vol.min()))
            verts, faces, _, _ = measure.marching_cubes_lewiner(
                vol, isovalue, spacing=(self.xstep, self.ystep, self.zstep)
            )
            ax.plot_trisurf(
                verts[:, 0],
                verts[:, 1],
                faces,
                verts[:, 2],
                cmap="Spectral",
                lw=1,
                alpha=1 - ival * 1.5,
            )

        plt.show()

    def planeplot_3D(self, cut):
        """ Plots dispersion relation as two-dimensional planes for xy, xz and yz."""
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm

        fig = plt.figure(figsize=plt.figaspect(0.3))
        ax1 = fig.add_subplot(1, 3, 1, projection="3d")
        xy = cut[cut[:, 2] == 0]
        x = xy[:, 0]
        y = xy[:, 1]
        z = xy[:, 3]
        surf = ax1.plot_trisurf(x, y, z, cmap=cm.plasma, linewidth=0.1)
        ax1.set_title("xy plane")

        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        xz = cut[cut[:, 1] == 0]
        x = xz[:, 0]
        y = xz[:, 2]
        z = xz[:, 3]
        surf2 = ax2.plot_trisurf(x, y, z, cmap=cm.plasma, linewidth=0.1)
        ax2.set_title("xz plane")

        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        xz = cut[cut[:, 0] == 0]
        x = xz[:, 1]
        y = xz[:, 2]
        z = xz[:, 3]
        surf3 = ax3.plot_trisurf(x, y, z, cmap=cm.plasma, linewidth=0.1)
        ax3.set_title("yz plane")

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def axplot_3D(self, fitparameters, fitfunction, data):
        """ Plots fit quality along main axes and diagonals in 3D."""
        f = fitfunction
        masses = [x for x in fitparameters]
        fig = plt.figure()
        ax = dict()
        for i in range(1, 10):
            ax[i] = fig.add_subplot(3, 3, i)

        dl = np.linspace(-self.scale, self.scale, 25)

        #### xx
        xdata = [[x, v] for x, y, z, v in data if (abs(y) < 1e-5 and abs(z) < 1e-5)]
        vl = np.fromiter((f(x, 0, 0, *masses) for x in dl * self.xstep), np.float)
        ax[1].plot(*zip(*sorted(xdata)), ls="", marker="o", label="data")
        ax[1].plot(dl * self.xstep, vl, color="crimson", label="fit")
        ax[1].set_ylabel(r"E - E$_F$ [a. u.]")
        ax[1].set_xlabel(r"$\Delta k_{xx}$ [a. u.]")
        ax[1].set_title("xx direction")

        #### xy
        xydata = [
            [x, v] for x, y, z, v in data if (abs(x - y) < 1e-5 and abs(z) < 1e-5)
        ]
        vl = np.fromiter((f(xy, xy, 0, *masses) for xy in dl * self.xstep), np.float)
        ax[2].plot(*zip(*sorted(xydata)), ls="", marker="o", label="data")
        ax[2].plot(dl * self.xstep, vl, color="crimson", label="fit")
        ax[2].set_ylabel(r"E - E$_F$ [a. u.]")
        ax[2].set_xlabel(r"$\Delta k_{xy}$ [a. u.]")
        ax[2].set_title("xy direction")

        #### xz
        xzdata = [
            [x, v] for x, y, z, v in data if (abs(x - z) < 1e-5 and abs(y) < 1e-5)
        ]
        vl = np.fromiter((f(xz, 0, xz, *masses) for xz in dl * self.xstep), np.float)
        ax[3].plot(*zip(*sorted(xzdata)), ls="", marker="o", label="data")
        ax[3].plot(dl * self.xstep, vl, color="crimson", label="fit")
        ax[3].set_ylabel(r"E - E$_F$ [a. u.]")
        ax[3].set_xlabel(r"$\Delta k_{xz}$ [a. u.]")
        ax[3].set_title("xz direction")

        #### yx
        yxdata = [
            [y, v] for x, y, z, v in data if (abs(y - x) < 1e-5 and abs(z) < 1e-5)
        ]
        vl = np.fromiter((f(xy, xy, 0, *masses) for xy in dl * self.ystep), np.float)
        ax[4].plot(*zip(*sorted(xydata)), ls="", marker="o", label="data")
        ax[4].plot(dl * self.ystep, vl, color="crimson", label="fit")
        ax[4].set_ylabel(r"E - E$_F$ [a. u.]")
        ax[4].set_xlabel(r"$\Delta k_{yx}$ [a. u.]")
        ax[4].set_title("yx direction")

        #### yy
        ydata = [[y, v] for x, y, z, v in data if (abs(x) < 1e-5 and abs(z) < 1e-5)]
        vl = np.fromiter((f(0, y, 0, *masses) for y in dl * self.ystep), np.float)
        ax[5].plot(*zip(*sorted(ydata)), ls="", marker="o", label="data")
        ax[5].plot(dl * self.ystep, vl, color="crimson", label="fit")
        ax[5].set_ylabel(r"E - E$_F$ [a. u.]")
        ax[5].set_xlabel(r"$\Delta k_{yy}$ [a. u.]")
        ax[5].set_title("yy direction")

        #### yz
        yzdata = [
            [y, v] for x, y, z, v in data if (abs(y - z) < 1e-5 and abs(x) < 1e-5)
        ]
        vl = np.fromiter((f(0, yz, yz, *masses) for yz in dl * self.ystep), np.float)
        ax[6].plot(*zip(*sorted(yzdata)), ls="", marker="o", label="data")
        ax[6].plot(dl * self.ystep, vl, color="crimson", label="fit")
        ax[6].set_ylabel(r"E - E$_F$ [a. u.]")
        ax[6].set_xlabel(r"$\Delta k_{yz}$ [a. u.]")
        ax[6].set_title("yz direction")

        #### xyz diagonal
        ddata = [
            [z, v] for x, y, z, v in data if (abs(x - y) < 1e-5 and abs(x - z) < 1e-5)
        ]
        vl = np.fromiter((f(d, d, d, *masses) for d in dl * self.zstep), np.float)
        ax[7].plot(*zip(*sorted(ddata)), ls="", marker="o", label="data")
        ax[7].plot(dl * self.zstep, vl, color="crimson", label="fit")
        ax[7].set_ylabel(r"E - E$_F$ [a. u.]")
        ax[7].set_xlabel(r"$\Delta k_{xyz}$ [a. u.]")
        ax[7].set_title("xyz diagonal")

        #### xy-z diagonal
        ddata = [
            [z, v] for x, y, z, v in data if (abs(x - y) < 1e-5 and abs(x + z) < 1e-5)
        ]
        vl = np.fromiter((f(d, d, -d, *masses) for d in dl * self.zstep), np.float)
        ax[8].plot(*zip(*sorted(ddata)), ls="", marker="o", label="data")
        ax[8].plot(dl * self.zstep, vl, color="crimson", label="fit")
        ax[8].set_ylabel(r"E - E$_F$ [a. u.]")
        ax[8].set_xlabel(r"$\Delta k_{xy-z}$ [a. u.]")
        ax[8].set_title("xy -z diagonal")

        #### zz
        zdata = [[z, v] for x, y, z, v in data if (abs(x) < 1e-4 and abs(y) < 1e-4)]
        vl = np.fromiter((f(0, 0, z, *masses) for z in dl * self.zstep), np.float)
        ax[9].plot(*zip(*sorted(zdata)), ls="", marker="o", label="data")
        ax[9].plot(dl * self.zstep, vl, color="crimson", label="fit")
        ax[9].set_ylabel(r"E - E$_F$ [a. u.]")
        ax[9].set_xlabel(r"$\Delta k_{zz}$ [a. u.]")
        ax[9].set_title("zz direction")
        ax[9].legend()

        plt.tight_layout()
        plt.show()

    def pick_band(self, point, spin=0):
        """ Picks closest energies, k-point and band index for given point an spin.
        
        Returns:
            tuple : closest k-point (ndarray), VBM energy (float), VBM band index (int), CBM energy (float), CBM band index (int)
        """

        def closest_point(point, points):
            tree = spatial.KDTree(points)
            idx = tree.query(point)[1]
            return idx

        idx = closest_point(point, self.kpoints)
        closest_k = self.kpoints[idx]
        VBM_at_site = np.max(
            self.eigenvalues[spin, idx, :][self.eigenvalues[spin, idx, :] < 0]
        )
        CBM_at_site = np.min(
            self.eigenvalues[spin, idx, :][self.eigenvalues[spin, idx, :] > 0]
        )
        VBM_band = np.argwhere(
            np.isclose(
                self.eigenvalues[spin, idx, :], VBM_at_site, rtol=1e-5, atol=1e-7
            )
        )
        CBM_band = np.argwhere(
            np.isclose(
                self.eigenvalues[spin, idx, :], CBM_at_site, rtol=1e-5, atol=1e-7
            )
        )
        VBM_band = [max(x) for x in VBM_band][0]
        CBM_band = [min(x) for x in CBM_band][0]
        logging.info(
            "Found valence band energy of {} Hartree for band {} at point k = ( {: 10.6f} {: 10.6f} {: 10.6f})".format(
                VBM_at_site, VBM_band, *closest_k
            )
        )
        logging.info(
            "Found conduction band energy of {} Hartree for band {} at point k = ( {: 10.6f} {: 10.6f} {: 10.6f})".format(
                CBM_at_site, CBM_band, *closest_k
            )
        )
        return closest_k, VBM_at_site, VBM_band, CBM_at_site, CBM_band

