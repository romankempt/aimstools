import aimstools
from aimstools.misc import *
from aimstools.structuretools import Structure

import numpy as np
import scipy.linalg
import scipy.spatial
import matplotlib
import matplotlib.pyplot as plt

from ase.dft.kpoints import bandpath, resolve_kpt_path_string

import ase.io
from ase.atoms import Atoms
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


class BrillouinZone:
    """ Mostly taken from ase.dft.bz, just cleaned up. """

    def __init__(
        self, atoms, bandpathstring: str = None, special_points: dict = None
    ) -> None:
        if isinstance(atoms, Structure):
            self.structure = atoms
        elif isinstance(atoms, (Atoms,)):
            self.structure = Structure(atoms)
        else:
            try:
                atoms = ase.io.read("atoms")
                self.structure = atoms
            except Exception:
                logger.critical(
                    "Could not parse structure from file or another atoms object."
                )
        self._special_points = special_points
        self._is_2d = self.structure.is_2d()
        self._set_bandpath(bandpathstring)

    def __repr__(self):
        return "{}(structure={}, is_2d={})".format(
            self.__class__.__name__, repr(self.structure), self._is_2d
        )

    @property
    def bandpath(self):
        return self._bandpath

    @property
    def special_points(self):
        return self._special_points

    @property
    def is_2d(self):
        return self._is_2d

    def _set_bandpath(self, bandpathstring=None):
        if bandpathstring == None:
            pbc = [1, 1, 1] if not self.is_2d else [1, 1, 0]
            bandpath = self.structure.cell.get_bravais_lattice(pbc=pbc).bandpath()
            self._special_points = bandpath.special_points
            bandpathstring = bandpath.path

        bp = BandPath(
            path=bandpathstring,
            cell=self.structure.cell,
            special_points=self.special_points,
        )
        self._bandpath = bp
        self._special_points = bp.special_points

    def _get_bz_vertices(self):
        atoms = self.structure
        cell = atoms.get_cell()
        icell = cell.reciprocal() * 2 * np.pi

        if self.is_2d:
            icell[2, 2] = 1e-3

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

    def plot(self, axes=None, azim=None, elev=None, **kwargs):
        special_points = self.special_points
        cell = self.structure.get_cell()
        icell = cell.reciprocal() * 2 * np.pi
        labelseq, coords = resolve_kpt_path_string(self.bandpath.path, special_points)
        paths = []
        points_already_plotted = set()
        for subpath_labels, subpath_coords in zip(labelseq, coords):
            subpath_coords = np.array(subpath_coords)
            subpath_coords = np.dot(subpath_coords, icell)
            points_already_plotted.update(subpath_labels)
            paths.append((subpath_labels, subpath_coords))

        with AxesContext(ax=axes, projections=[["3d"]], **kwargs) as axes:
            assert axes.name == "3d", "Axes must be 3D."
            azim = np.pi if self.is_2d else azim
            elev = np.pi / 2 if self.is_2d else elev
            self._plot_3d_bz(axes=axes, paths=paths, azim=azim, elev=elev)
        return axes

    def _plot_3d_bz(self, axes, paths, azim=None, elev=None):
        azim = azim or np.pi / 5
        elev = elev or np.pi / 6
        x = np.sin(azim)
        y = np.cos(azim)
        view = [x * np.cos(elev), y * np.cos(elev), np.sin(elev)]
        maxp = 0.0
        minp = 0.0
        bz = self._get_bz_vertices()
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
            axes.plot(x, y, z, c="tab:blue", ls="-", marker=".", zorder=1)
            for name, point in zip(names, points):
                x, y, z = point
                name = pretty(name)
                axes.scatter(x, y, z, c="tab:red", s=2, zorder=2)
                axes.text(
                    x, y, z, name, ha="center", va="bottom", color="tab:red", zorder=3
                )
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
