from AIMS_tools.misc import *
from AIMS_tools.structuretools import structure as strc

import spglib

import numpy as np
import ase.build


def _coincidence(a1, b1, m1, m2, n1, n2, theta):
    theta = theta / 180.0 * np.pi
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    Am = a1 @ np.array([m1, m2])
    Bn = R @ b1 @ np.array([n1, n2])
    return (m1, m2, n1, n2, theta, np.linalg.norm(Am - Bn))


def _standardize(collection, symprec=1e-4):
    atoms = collection.atoms
    cell = (atoms.get_cell()).tolist()
    pos = atoms.get_scaled_positions().tolist()
    numbers = atoms.get_atomic_numbers()

    cell, scaled_pos, numbers = spglib.standardize_cell(
        (cell, pos, numbers), to_primitive=True, symprec=symprec, no_idealize=True
    )

    atoms = ase.atoms.Atoms(
        scaled_positions=scaled_pos, numbers=numbers, cell=cell, pbc=True
    )
    axes = [0, 1, 2]
    lengths = atoms.cell.lengths()
    order = [x for x, y in sorted(zip(axes, lengths), key=lambda pair: pair[1])]
    if order != [0, 1, 2]:
        atoms = ase.geometry.permute_axes(atoms, order)
    collection._replace(atoms=atoms)
    return collection


class interface:
    """Interface class to construct heterojunctions from two layered structures.

    Employs the algorithm outlined by Schwalbe-Koda (J. Phys. Chem. C 2016, 120, 20, 10895-10908) to find coincidence lattices of 2D crystals.

    Note:
        Employs parallelism through the multiprocessing library. Hence, the scaling is roughly :math:`(N_{trans})^4 \cdot N_{angles} / N_{cpus}`.

    Example:
        >>> from AIMS_tools.interfaces import interface
        >>> intf = interface(bottom, top)
        >>> intf.analyze_results(weight=0.5, distance=4)
        >>> intf.plot_results(jitter=0.05)
    
    Args:
        bottom (atoms): Input structure for the lower layer. Can be path to file, ase.atoms.Atoms object or AIMS_tools.structuretools.structure object.
        top (atoms): Input structure for the upper layer. Can be path to file, ase.atoms.Atoms object or AIMS_tools.structuretools.structure object.
    
    Keyword Args:
        N_translations (int): Number of translations or maximum super cell size to consider, defaults to 10.
        angle_stepsize (float): Stepsize between angles to rotate top layer. Defaults to 10.0.
        angle_limits (tuple): Lower and upper limit of angles in degrees to scan. Defaults to (0, 180).
        crit (float): Distance criterion to accept coincidence lattice points. Defaults to 0.05 (Angström).
        distance (float): Interlayer distance between the reconstructed stacks. Defaults to 4.0 (Angström).
        weight (float): Value between 0 and 1, defaults to 0.5. The unit cell of the reconstructed stack is :math:`B + w \cdot (T - B)`.
        prec (float): Precision to identify equivalent structures. Defaults to 1e-4.
        jitter (float): Jitters data points for plotting for easier picking. Defaults to 0.05.
    
    Attributes:
        results (dict): Dictionary of angles in radians and list of lists containing (m1, m2, n1, n2, err).
        pairs (dict): Dictionary of angles in degrees and tuples of supercell matrices (M, N).
        solved (list): List of stacks as named tuples after analyzing the results. Contains (atoms, M, N, angle, stress).

    """

    def __init__(self, bottom, top, **kwargs):
        a1, b1 = self.__checks_and_setup(bottom, top)
        N_translations = kwargs.get("N_translations", 10)
        angle_stepsize = kwargs.get("angle_stepsize", 10.0)
        angle_limits = kwargs.get("angle_limits", (0, 180))
        crit = kwargs.get("crit", 0.1)
        distance = kwargs.get("distance", 4.0)
        self.weight = kwargs.get("weight", 0.5)
        prec = kwargs.get("prec", 1e-4)
        jitter = kwargs.get("jitter", 0.05)

        self.results = self.find_coincidence(
            a1,
            b1,
            N=N_translations,
            stepsize=angle_stepsize,
            crit=crit,
            angle_limits=angle_limits,
        )
        if self.results != None:
            self.pairs = self.find_noncollinear_pairs()

    def __checks_and_setup(self, bottom, top):
        if type(bottom) == ase.atoms.Atoms:
            self.bottom = strc(bottom.copy())
        elif Path(bottom).is_file():
            self.bottom = strc(Path(bottom))
        elif type(bottom) == strc:
            self.bottom = bottom
        else:
            logging.critical("Input for bottom structure not recognised.")
        if type(top) == ase.atoms.Atoms:
            self.top = strc(top.copy())
        elif Path(bottom).is_file():
            self.top = strc(Path(top))
        elif type(top) == strc:
            self.top = top
        else:
            logging.critical("Input for top structure not recognised.")
        assert self.bottom.is_2d(self.bottom.atoms), "Bottom structure is not 2D!"
        assert self.top.is_2d(self.top.atoms), "Top structure is not 2D!"
        self.bottom.standardize(to_primitive=True)
        self.top.standardize(to_primitive=True)
        # self.bottom.set_cell(self.bottom.cell.T, scale_atoms=True)
        # self.top.set_cell(self.top.cell.T, scale_atoms=True)
        a1 = self.bottom.atoms.cell[[0, 1], :2].T.copy()
        b1 = self.top.atoms.cell[[0, 1], :2].T.copy()
        return (a1, b1)

    def find_coincidence(
        self,
        a1,
        b1,
        N=10,
        stepsize=1,
        crit=0.05,
        angle_limits=(0, 180.0),
        chunksize=10000,
    ):
        """ Solves the system of linear equations given by :math:`\mathbf{A m = R B n}` employing multiprocessing.
    
        Args:
            a1 (ndarray): (2, 2) ndarray representing lower basis.
            b1 (ndarray): (2, 2) ndarray representing upper basis.
            N (int, optional): Maximum supercell size. Defaults to 10.
            stepsize (int, optional): Angular step size. Defaults to 1.
            crit (float, optional): Acceptance criterion. Defaults to 0.05.
            angle_limits (tuple, optional): Lower and upper limit of angular range. Defaults to (0, 180.0).
            chunksize (int): Distributed chunks in the asynchronous pool mapping.
        
        Returns:
            dict: Pairs of angle : (m1, m2, n1, n2, err).

        """

        import multiprocessing as mp
        import itertools

        start = time.time()
        cpus = mp.cpu_count()
        pool = mp.Pool(processes=cpus)
        nrange = range(-N + 1, N)
        angles = np.arange(angle_limits[0], angle_limits[1], stepsize)
        iterator = (
            (a1, b1, m1, m2, n1, n2, theta)
            for m1, m2, n1, n2, theta in itertools.product(
                nrange, nrange, nrange, nrange, angles
            )
        )
        ngrids = len(nrange) ** 4 * len(angles)
        logging.info("Initializing {} gridpoints ...".format(ngrids))
        logging.info("Parallelising on {} processors ...".format(cpus))
        res = pool.starmap_async(_coincidence, iterator, chunksize=chunksize)
        res = res.get()
        pool.close()
        pool.join()
        results = np.zeros((len(res), 6))
        for i, j in enumerate(res):
            results[i] = j
        index = np.argwhere(
            (
                ((results[:, 0] == 0) & (results[:, 1] == 0))
                | ((results[:, 2] == 0) & (results[:, 3] == 0))
            )
            == False
        )
        results = results[index[:, 0]]
        index = np.argwhere(results[:, 5] < crit)[:, 0]
        mins = results[index]
        end = time.time()
        logging.info(
            "Scanning angular and translational grid finished in {:.2f} seconds.".format(
                end - start
            )
        )
        if mins.size != 0 or mins.size == 1:
            logging.info("Found {:d} matching lattice points.".format(len(mins)))
            d = {}
            for row in mins:
                m1, m2, n1, n2, theta, diff = row
                if theta in d.keys():
                    d[theta].append([m1, m2, n1, n2, diff])
                elif theta not in d.keys():
                    d[theta] = [[m1, m2, n1, n2, diff]]
            return d
        else:
            logging.error("Found no matching lattice points.")
            return None

    def find_noncollinear_pairs(self):
        r""" Finds linearly independent pairs of :math:`(\vec{m}_1, \vec{m}_2)` and :math:`(\vec{n}_1, \vec{n}_2)`.

        Constructs supercell matrices :math:`M=(\vec{m}_1^T, \vec{m}_2^T)` and :math:`N=(\vec{n}_1^T, \vec{n}_2^T)` with positive determinants.
        Collinear matrices at the same angle are filtered such that the area of the unit cell :math:`(\vec{m}_1 \times \vec{m}_2)` is minimized.

        
        Returns:
            dict: Pairs of angle : (M, N).

        """
        results = self.results
        d = {}
        for ang, indices in results.items():
            for i, pair1 in enumerate(indices):
                pairs = []
                angle = ang * 180.0 / np.pi
                m1, m2, n1, n2, stress = pair1
                for k, pair2 in enumerate(indices):
                    if k > i:
                        m3, m4, n3, n4, _ = pair2
                        M = np.abs(np.cross([m1, m2], [m3, m4])) < 1e-5
                        N = np.abs(np.cross([n1, n2], [n3, n4])) < 1e-5
                        dependent = M and N
                        if not dependent:
                            M = np.array([[m1, m2, 0], [m3, m4, 0], [0, 0, 1]])
                            N = np.array([[n1, n2, 0], [n3, n4, 0], [0, 0, 1]])
                            if (np.linalg.det(M) > 0) and (np.linalg.det(N) > 0):
                                pairs.append((M, N))
                if angle in d.keys():
                    d[angle] += pairs
                elif pairs != []:
                    d[angle] = pairs
        s = 0
        for k, v in d.items():
            s += len(v)
        logging.info(
            "Constructed {:d} linearly independent pairs of supercell matrices.".format(
                s
            )
        )
        fd = {}
        for ang, pairs in d.items():
            collinear = []
            for i, p1 in enumerate(pairs):
                M1, N1 = p1
                for j, p2 in enumerate(pairs):
                    if j > i:
                        M2, N2 = p2
                        c1 = np.linalg.norm(np.cross(M1[:, 0], M2[:, 0])) < 1e-5
                        c2 = np.linalg.norm(np.cross(M1[:, 1], M2[:, 1])) < 1e-5
                        coll = c1 and c2
                        if coll:
                            ar1 = np.linalg.norm(np.cross(M1[:, 0], M1[:, 1]))
                            ar2 = np.linalg.norm(np.cross(M2[:, 0], M2[:, 1]))
                            if ar2 > ar1:
                                collinear.append(j)
                            elif ar1 > ar2:
                                collinear.append(i)

            fd[ang] = [k for v, k in enumerate(d[ang]) if v not in collinear]
        s = 0
        for k, v in fd.items():
            s += len(v)
        logging.info("Reduced to {:d} non-collinear matrices.".format(s))
        return fd

    def analyze_results(
        self, distance=3, weight=0.5, prec=1e-4,
    ):
        """ Analyzes results after grid point search.

        All structures are standardized via spglib to the primitive unit with enforced axis order :math:`a < b < c`.
        Structures are considered identical if they have the same number of atoms, similar unit cell area and stress on the lattice vectors.
        
        Args:
            distance (int, optional): Interlayer distance of reconstructed stacks. Defaults to 3.
            weight (float, optional): Value between 0 and 1, defaults to 0.5. The unit cell of the reconstructed stack is :math:`B + w \cdot (T - B)`.
            prec ([type], optional): Precision to identify equivalent structures. Defaults to 1e-4.
        
        Note:
            The ASE algorithm to generate supercells might not work in some cases. Please note and report these.

        """
        from collections import namedtuple

        pairs = self.pairs
        self.weight = weight
        assert (self.weight <= 1.0) and (
            self.weight >= 0.0
        ), "Weight must be between 0 and 1."

        data = namedtuple("stack", ["atoms", "M", "N", "angle", "stress"])
        solved = []
        start = time.time()
        logging.info("Reconstructing heterostructures ...")
        for angle, indices in pairs.items():
            for row in indices:
                bottom = self.bottom.atoms.copy()
                top = self.top.atoms.copy()
                M, N = row
                try:
                    bottom = ase.build.make_supercell(bottom, M, wrap=False)
                    top.rotate(angle, v="z", rotate_cell=True)
                    top = ase.build.make_supercell(top, N, wrap=False)
                    bottom = strc(bottom).recenter(bottom)
                    top = strc(top).recenter(top)
                    stack = self.stack(
                        bottom, top, distance=distance, weight=self.weight
                    )
                    stress = np.linalg.norm(top.cell - bottom.cell)
                    solved.append(data(stack, M, N, angle, stress))
                except:
                    logging.error(
                        "ASE supercell generation didn't work for \n{} and \n{}".format(
                            M, N
                        )
                    )
        logging.info(
            "Standardizing representations and filtering unique structures ..."
        )
        self.__filter_unique_structures(solved, prec=prec)
        end = time.time()
        logging.info("Analysis finished in {:.2f} seconds.".format(end - start))

    def stack(self, bottom, top, weight=0.5, distance=4):
        """ Stacks two layered structures on top of each other.
        
        Args:
            bottom (atoms): Lower layer.
            top (atoms): Upper layer.
            weight (float, optional): Value between 0 and 1, defaults to 0.5. The unit cell of the reconstructed stack is :math:`B + w \cdot (T - B)`.
            distance (int, optional): Interlayer distance in Angström. Defaults to 4.
        
        Returns:
            atoms: Reconstructed stack.

        """

        bottom = bottom.copy()
        top = top.copy()
        c1 = np.linalg.norm(bottom.cell[2])
        c2 = np.linalg.norm(top.cell[2])
        cell1 = bottom.cell.copy()
        cell2 = top.cell.copy()
        cell1[2] /= c1
        cell2[2] /= c2
        cell = cell1 + weight * (cell2 - cell1)
        cell[2] /= np.linalg.norm(cell[2])
        cell1 = cell.copy()
        cell2 = cell.copy()
        cell1[2] *= c1
        cell2[2] *= c2

        bottom.set_cell(cell1, scale_atoms=True)
        top.set_cell(cell2, scale_atoms=True)

        zeroshift = np.min(bottom.get_positions()[:, 2])
        bottom.translate([0, 0, -zeroshift])
        zeroshift = np.min(top.get_positions()[:, 2])
        top.translate([0, 0, -zeroshift])
        bottom_thickness = np.max(bottom.get_positions()[:, 2]) - np.min(
            bottom.get_positions()[:, 2]
        )
        top.translate([0, 0, bottom_thickness])
        top.translate([0, 0, distance])
        bottom.extend(top)
        stack = strc(bottom).recenter(bottom)
        return stack

    def __spglib_standard(self, solved, prec):
        start = time.time()
        solved = [_standardize(j, prec) for j in solved]
        end = time.time()

        logging.info(
            "  Spglib standardization with enforced axes order finished after {:.2f} seconds ...".format(
                end - start
            )
        )
        return solved

    def __remove_doubles(self, solved, prec):
        copies = []
        start = time.time()
        for i, k in enumerate(solved):
            for j, h in enumerate(solved):
                if j > i:
                    a = k.atoms.copy()
                    b = h.atoms.copy()
                    area1 = np.linalg.norm(np.cross(a.cell[:, 0], a.cell[:, 1]))
                    area2 = np.linalg.norm(np.cross(b.cell[:, 0], b.cell[:, 1]))
                    area = np.abs((area2 - area1)) < prec
                    natoms = (len(a) - len(b)) == 0
                    stress = np.abs(solved[j].stress - solved[i].stress) < prec
                    if area and natoms and stress:
                        copies.append(j)
        solved = [i for j, i in enumerate(solved) if j not in copies]
        end = time.time()
        logging.info(
            "  Filtering structures finished in {:.2f} seconds ...".format(end - start)
        )
        return solved

    def __filter_unique_structures(self, solved, prec=1e-4):
        solved = self.__spglib_standard(solved, prec)
        solved = self.__remove_doubles(solved, prec)
        logging.info("Found {:d} unique structures.".format(len(solved)))
        self.solved = solved

    def plot_results(self, jitter=0.05):
        """ Plots results interactively.

        Generates a matplotlib interface that allows to select the reconstructed stacks and save them to a file.
       
        Args:
            jitter (float, optional): Jitters data points to make picking easier. Defaults to 0.05.       

        """
        from matplotlib.widgets import Button

        def rand_jitter(arr, jitter):
            stdev = jitter * (max(arr) - min(arr))
            return arr + np.random.randn(len(arr)) * stdev

        data = np.array([[i.stress, len(i.atoms)] for i in self.solved], dtype=float)

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=([6.4 * 3, 4.8 * 1.5]))
        ax[0].scatter(
            rand_jitter(data[:, 0], jitter),
            data[:, 1],
            color="crimson",
            alpha=0.5,
            picker=3.5,
            marker=".",
        )
        ax[0].set_ylim(0, ax[0].get_ylim()[1])
        ax[0].set_xlabel("Distance between unit cells [Angström]")
        ax[0].set_ylabel("Number of atoms")
        ax[0].set_title("Click a point to select structure.")
        ax[0].grid(
            axis="both", color="lightgray", linestyle="-", linewidth=1, alpha=0.2
        )
        ax[1].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_xlabel("")
        ax[1].set_ylabel("")
        ax[1].set_frame_on(False)
        ax[2].set_yticks([])
        ax[2].set_xticks([])
        ax[2].set_xlabel("")
        ax[2].set_ylabel("")
        ax[2].set_frame_on(False)
        axbutton = plt.axes([0.8, 0.05, 0.1, 0.05])
        fig.canvas.mpl_connect("pick_event", self.__onpick)

        def __save(stack):
            try:
                name = "{}_M{}{}{}{}_N{}{}{}{}_a{:.2f}.xyz".format(
                    self.current_stack.get_chemical_formula(), *self.current_scdata[1:]
                )
                stack.write(name, vec_cell=True)
                logging.info("Saved structure to {}".format(name))
            except:
                logging.error("You need to select a point first.")

        save = Button(axbutton, " Save this structure. ")
        save.on_clicked(lambda x: __save(self.current_stack))
        plt.show()

    def __onpick(self, event):
        point = event.artist
        mouseevent = event.mouseevent
        # xdata = point.get_xdata()[0]
        # ydata = point.get_ydata()[0]
        # index = np.where((self.data[:, 0] == xdata) & (self.data[:, 1] == ydata))
        # index = index[0][0]
        index = event.ind[0]
        fig = point.properties()["figure"]
        axes = fig.axes
        stack = self.solved[index].atoms
        M = self.solved[index].M
        N = self.solved[index].N
        angle = self.solved[index].angle
        m1, m2, m3, m4 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        n1, n2, n3, n4 = N[0, 0], N[0, 1], N[1, 0], N[1, 1]
        scdata = (
            int(len(stack)),
            int(m1),
            int(m2),
            int(m3),
            int(m4),
            int(n1),
            int(n2),
            int(n3),
            int(n4),
            float(angle),
        )
        self.current_scdata = scdata
        self.__plot_stack(stack, fig, axes[2], scdata)
        basis1 = self.bottom.atoms.copy()
        basis2 = self.top.atoms.copy()
        basis2.rotate(scdata[-1], v="z", rotate_cell=True)
        self.__plot_lattice_points(
            fig, axes[1], basis1.cell, basis2.cell, scdata,
        )

    def __plot_stack(self, stack, fig, axes, scdata):
        from ase.visualize.plot import plot_atoms

        canvas = fig.canvas
        axes.clear()
        axes.set_yticks([])
        axes.set_xticks([])
        axes.set_xlabel("")
        axes.set_ylabel("")
        scdata = "{:d} atoms, twist angle of {:.2f}°".format(len(stack), scdata[-1])
        axes.set_title(scdata)
        self.current_stack = stack
        plot_atoms(stack, axes, radii=0.3)
        axes.set_frame_on(False)
        canvas.draw()

    def __plot_lattice_points(self, fig, axes, basis1, basis2, scdata, N=20):
        import itertools
        from matplotlib import path, patches

        canvas = fig.canvas
        axes.clear()
        axes.set_yticks([])
        axes.set_xticks([])
        axes.set_xlabel("")
        axes.set_ylabel("")

        sc1 = np.array(scdata[1:5]).reshape((2, 2))
        sc2 = np.array(scdata[5:9]).reshape((2, 2))

        # first cell
        A = basis1[[0, 1], :2]
        a1 = A[:, 0]
        a2 = A[:, 1]
        p = itertools.product(range(-N + 1, N), repeat=2)
        new = list()
        for n in p:
            point = n[0] * a1 + n[1] * a2
            new.append(point)
        Agrid = np.stack(new)
        axes.scatter(Agrid[:, 0], Agrid[:, 1], color="darkred", alpha=0.5, s=3)
        A = sc1 @ A
        path1 = [(0, 0), (A[:, 0]), (A[:, 0] + A[:, 1]), (A[:, 1]), (0, 0)]
        p = path.Path(path1)
        patch = patches.PathPatch(p, facecolor="red", lw=1, alpha=0.15)
        axes.add_patch(patch)

        # second cell
        B = basis2[[0, 1], :2]
        b1 = B[:, 0]
        b2 = B[:, 1]
        p = itertools.product(range(-N + 1, N), repeat=2)
        new = list()
        for n in p:
            point = n[0] * b1 + n[1] * b2
            new.append(point)
        Bgrid = np.stack(new)
        axes.scatter(Bgrid[:, 0], Bgrid[:, 1], color="darkblue", alpha=0.5, s=3)
        B = sc2 @ B
        path2 = [(0, 0), (B[:, 0]), (B[:, 0] + B[:, 1]), (B[:, 1]), (0, 0)]
        p = path.Path(path2)
        patch = patches.PathPatch(p, facecolor="darkblue", lw=1, alpha=0.15)
        axes.add_patch(patch)

        # supercell
        C = A + self.weight * (B - A)
        path3 = [(0, 0), (C[:, 0]), (C[:, 0] + C[:, 1]), (C[:, 1]), (0, 0)]
        p = path.Path(path3)
        patch = patches.PathPatch(
            p, facecolor="none", edgecolor="purple", lw=2, alpha=0.5
        )
        axes.add_patch(patch)
        path3 = np.array(path3)
        xlim = (np.min(path3[:, 0]) - 4, np.max(path3[:, 0] + 4))
        ylim = (np.min(path3[:, 1]) - 4, np.max(path3[:, 1] + 4))
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        axes.set_frame_on(False)
        natoms, m1, m2, m3, m4, n1, n2, n3, n4, angle = scdata
        scdata = """M = ({: 2d}, {: 2d}, {: 2d}, {: 2d})\nN = ({: 2d}, {: 2d}, {: 2d}, {: 2d})""".format(
            m1, m2, m3, m4, n1, n2, n3, n4
        )
        axes.set_title(scdata)
        canvas.draw()
