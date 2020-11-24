from aimstools.misc import *
from aimstools.structuretools import Structure

import numpy as np
import scipy.linalg
import scipy.spatial
import matplotlib.pyplot as plt

from ase.dft.kpoints import resolve_kpt_path_string

import ase.io
from ase.dft.kpoints import BandPath


def pretty(kpt):
    if kpt == "G":
        kpt = r"$\Gamma$"
    elif len(kpt) == 2:
        kpt = kpt[0] + "$_" + kpt[1] + "$"
    return kpt


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


class BrillouineZone:
    """ Mostly taken from ase.dft.bz, just cleaned up. """

    def __init__(self, atoms, bandpathstring, special_points: dict = None) -> None:
        if type(atoms) == ase.atoms.Atoms:
            self.structure = Structure(atoms)
        else:
            self.structure = atoms
        self.special_points = special_points
        self.__set_bandpath(bandpathstring)
        self.is_2d = self.structure.is_2d()

    def __repr__(self):
        return "{}(structure={}, is_2d={})".format(
            self.__class__.__name__, repr(self.structure), self.is_2d
        )

    @property
    def bandpath(self):
        return self._bandpath

    def __set_bandpath(self, bandpathstring):
        bp = BandPath(
            path=bandpathstring,
            cell=self.structure.cell,
            special_points=self.special_points,
        )
        self._bandpath = bp
        self.special_points = bp.special_points

    def get_bz(self, dim=3):
        atoms = self.structure
        cell = atoms.get_cell()
        icell = np.linalg.pinv(cell).T

        if dim < 3:
            icell[2, 2] = 1e-3
        if dim < 2:
            icell[1, 1] = 1e-3

        bz = []
        I = (np.indices((3, 3, 3)) - 1).reshape((3, 27))
        G = np.dot(icell.T, I).T
        voronoi = scipy.spatial.Voronoi(G)
        for vertices, points in zip(voronoi.ridge_vertices, voronoi.ridge_points):
            if -1 not in vertices and 13 in points:
                normal = G[points].sum(0)
                normal /= (normal ** 2).sum() ** 0.5
                bz.append((voronoi.vertices[vertices], normal))
        return bz

    def plot(self, axes=None, paths=None, points=None, elev=None):
        special_points = self.special_points
        labelseq, coords = resolve_kpt_path_string(self.bandpath.path, special_points)
        fig = plt.gcf()
        paths = []
        points_already_plotted = set()
        for subpath_labels, subpath_coords in zip(labelseq, coords):
            subpath_coords = np.array(subpath_coords)
            points_already_plotted.update(subpath_labels)
            paths.append((subpath_labels, self.bandpath._scale(subpath_coords)))

        dimensions = 2 if self.is_2d else 3
        if dimensions == 3:
            if axes == None:
                axes = fig.gca(projection="3d")
            try:
                axes = self.plot_3d_bz(axes=axes, paths=paths)
            except Exception as exct:
                print(exct)
                logger.error(
                    "An exception occured. Most likely you did not specify a 3D-projected axis, but your brillouine zone is 3D."
                )
        elif dimensions == 2:
            if axes == None:
                axes = fig.gca()
            axes = self.plot_2d_bz(axes=axes, paths=paths)
        else:
            raise Exception(
                "1D and 0D Brillouine zones do not make much sense to plot."
            )
        return axes

    def plot_3d_bz(self, axes, paths, elev=None):
        azim = np.pi / 5
        elev = elev or np.pi / 6
        x = np.sin(azim)
        y = np.cos(azim)
        view = [x * np.cos(elev), y * np.cos(elev), np.sin(elev)]
        maxp = 0.0
        minp = 0.0
        bz = self.get_bz(dim=3)
        for points, normal in bz:
            x, y, z = np.concatenate([points, points[:1]]).T
            maxp = max(maxp, points.max())
            minp = min(minp, points.min())

            if np.dot(normal, view) < 0:
                ls = ":"
            else:
                ls = "-"
            axes.plot(x, y, z, c=mutedblack, ls=ls)
        for names, points in paths:
            x, y, z = np.array(points).T
            axes.plot(x, y, z, c="royalblue", ls="-", marker=".")
            for name, point in zip(names, points):
                x, y, z = point
                name = pretty(name)
                axes.text(x, y, z, name, ha="center", va="bottom", color="crimson")
        axes.set_axis_off()
        axes.view_init(azim=azim / np.pi * 180, elev=elev / np.pi * 180)

        if hasattr(axes, "set_proj_type"):
            axes.set_proj_type("ortho")
        set_axes_equal(axes)
        minp0 = 0.9 * minp  # Here we cheat a bit to trim spacings
        maxp0 = 0.9 * maxp
        axes.set_xlim3d(minp0, maxp0)
        axes.set_ylim3d(minp0, maxp0)
        axes.set_zlim3d(minp0, maxp0)

        return axes

    def plot_2d_bz(self, axes, paths):
        cell = self.structure.cell.copy()
        assert all(abs(cell[2][0:2]) < 1e-6) and all(abs(cell.T[2][0:2]) < 1e-6)

        points = self.bandpath._scale(self.bandpath.kpts)
        bz = self.get_bz(dim=2)

        maxp = 0.0
        minp = 0.0
        for points, _ in bz:
            x, y, z = np.concatenate([points, points[:1]]).T
            axes.plot(x, y, c=mutedblack, ls="-")
            maxp = max(maxp, points.max())
            minp = min(minp, points.min())

        for names, points in paths:
            x, y, z = np.array(points).T
            axes.plot(x, y, c="royalblue", ls="-")

            for name, point in zip(names, points):
                x, y, z = point
                name = pretty(name)

                ha_s = ["right", "left", "right"]
                va_s = ["bottom", "bottom", "top"]

                ha = ha_s[int(np.sign(x))]
                va = va_s[int(np.sign(y))]
                if abs(z) < 1e-6:
                    axes.text(
                        x, y, name, ha=ha, va=va, color="crimson", zorder=5,
                    )

        axes.set_axis_off()
        axes.autoscale_view(tight=True)
        s = maxp * 1.05
        axes.set_xlim(-s, s)
        axes.set_ylim(-s, s)
        axes.set_aspect("equal")

        return axes

