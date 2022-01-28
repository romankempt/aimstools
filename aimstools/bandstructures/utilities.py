from aimstools.misc import *

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.colors import Normalize

from collections import namedtuple

import numpy as np

from scipy import interpolate


class DirectBandGap:
    """Container class to store information about the direct band gap."""

    def __init__(
        self,
        value: float = None,
        spin_index: int = None,
        k_index: int = None,
        k_axis_coords: float = None,
        kpoint: list = None,
        vbm: float = None,
        cbm: float = None,
    ) -> None:
        self.value = value
        self.spin = spin_index
        self.k_index = k_index
        self.k_axis_coords = k_axis_coords
        self.kpoint = kpoint
        self.vbm = vbm
        self.cbm = cbm
        self.__check()

    def __check(self):
        assert self.value >= 0.0, "Band gap value cannot be negative."
        assert self.spin in [0, 1], "Spin index must be 0 or 1."
        assert len(self.kpoint) == 3, "Fractional k-point coordinates must be length 3."
        assert not any(
            l > 1.0 for l in self.kpoint
        ), "k-point coordinates must be fractional."
        assert isinstance(
            self.k_axis_coords, (np.floating, float)
        ), "Coordinate on k-axis must be float."

    def __repr__(self):
        return "{}(value={}, kpoint={})".format(
            self.__class__.__name__, self.value, self.kpoint
        )


class IndirectBandGap:
    """Container class to store information about the indirect band gap."""

    def __init__(
        self,
        value: float = None,
        spin_index: int = None,
        k_index1: int = None,
        k_index2: int = None,
        k_axis_coords1: float = None,
        k_axis_coords2: float = None,
        kpoint1: list = None,
        kpoint2: list = None,
        vbm: float = None,
        cbm: float = None,
    ) -> None:
        self.value = value
        self.spin = spin_index
        self.k_index1 = k_index1
        self.k_index2 = k_index2
        self.k_axis_coords1 = k_axis_coords1
        self.k_axis_coords2 = k_axis_coords2
        self.kpoint1 = kpoint1
        self.kpoint2 = kpoint2
        self.vbm = vbm
        self.cbm = cbm
        self.__check()

    def __check(self):
        assert self.value >= 0.0, "Band gap value cannot be negative."
        assert self.spin in [0, 1], "Spin index must be 0 or 1."
        assert (
            len(self.kpoint1) == 3
        ), "Fractional k-point coordinates must be length 3."
        assert (
            len(self.kpoint2) == 3
        ), "Fractional k-point coordinates must be length 3."
        assert not any(
            l > 1.0 for l in self.kpoint1
        ), "k-point coordinates must be fractional."
        assert not any(
            l > 1.0 for l in self.kpoint2
        ), "k-point coordinates must be fractional."
        assert isinstance(
            self.k_axis_coords1, (np.floating, float)
        ), "Coordinate on k-axis must be float."
        assert isinstance(
            self.k_axis_coords2, (np.floating, float)
        ), "Coordinate on k-axis must be float."

    def __repr__(self):
        return "{}(value={}, kpoint_vbm={}, kpoint_cbm={})".format(
            self.__class__.__name__, self.value, self.kpoint1, self.kpoint2
        )


class BandSpectrum:
    """Container class for eigenvalue spectrum and associated data.

    Attributes:
        atoms (ase.atoms.Atoms): ASE atoms object.
        kpoints (ndarray): (nkpoints, 3) array with k-points.
        kpoint_axis (ndarray): (nkpoints, 1) linear plotting axis.
        eigenvalues (ndarray): (nkpoints, nbands) array with eigenvalues in eV.
        occupations (ndarray): (nkpoints, nbands) array with occupations.
        contributions (MullikenContribution): :class:`~aimstools.bandstructures.mulliken_bandstructure.MullikenContribution`.
        label_coords (list): List of k-point label coordinates on the plotting axis.
        kpoint_labels (list): List of k-point labels.
        jumps (list): List of jumps from unconnected Brillouin zone sections on the plotting axis.
        fermi_level (float): Fermi level in eV.
        reference (str): Reference energy description.
        band_extrema (tuple): Band extrema (VBM, CBM) from the FHI-aims output file.
        shift (float): Value to shift energies for reference.
        bandpath (str): Bandpath string in ASE format.


    """

    def __init__(
        self,
        atoms: "ase.atoms.Atoms" = None,
        kpoints: "numpy.ndarray" = None,
        kpoint_axis: "numpy.ndarray" = None,
        eigenvalues: "numpy.ndarray" = None,
        occupations: "numpy.ndarray" = None,
        label_coords: list = None,
        kpoint_labels: list = None,
        jumps: list = None,
        fermi_level: float = None,
        reference: str = None,
        band_extrema: tuple = None,
        shift: float = None,
        bandpath: str = None,
    ) -> None:
        self._atoms = atoms
        self._kpoints = kpoints
        self._kpoint_axis = kpoint_axis
        self._eigenvalues = eigenvalues
        self._occupations = occupations
        self._label_coords = label_coords
        self._kpoint_labels = kpoint_labels
        self._jumps = jumps
        self._fermi_level = fermi_level
        self._reference = reference
        self._band_extrema = band_extrema
        self._shift = shift
        self._bandpath = bandpath
        self._bandgap = None

    def _find_direct_gap(self):
        nk, ns, nb = self.eigenvalues.shape
        evs = self.eigenvalues.copy()
        occs = self.occupations.copy()
        kpts = self.kpoints.copy()
        kcoords = self.kpoint_axis.copy()

        results = []
        for s in range(ns):
            gaps = []
            for k in range(nk):
                vbs = evs[k, s, :][occs[k, s, :] >= 1e-4]
                cbs = evs[k, s, :][occs[k, s, :] < 1e-4]
                cb = np.min(cbs)
                vb = np.max(vbs)
                gap = cb - vb
                gaps.append((k, gap, vb, cb))
            gaps = np.array(gaps)
            index = np.argmin(gaps[:, 1])
            if gaps[index, 1] < 0.1:
                # probably metallic along this spin channel
                results.append(None)
            if gaps[index, 1] >= 0.1:
                k, value, vbm, cbm = gaps[index]
                k = int(k)
                kp = np.dot(kpts[k, :], self.atoms.cell.T) / (2 * np.pi)
                kc = kcoords[k]
                direct = DirectBandGap(
                    value=value,
                    spin_index=s,
                    k_index=k,
                    k_axis_coords=kc,
                    kpoint=kp,
                    cbm=cbm,
                    vbm=vbm,
                )
                results.append(direct)
        return results

    def get_direct_gap(self, spin):
        """ Returns direct band gap for given spin channel or None if metallic along this spin channel."""
        results = self._find_direct_gap()
        s = self._spin2index(spin)
        return results[s]

    def _find_indirect_gap(self):
        from itertools import combinations

        nk, ns, nb = self.eigenvalues.shape
        evs = self.eigenvalues.copy()
        occs = self.occupations.copy()
        kpts = self.kpoints.copy()
        kcoords = self.kpoint_axis.copy()

        results = []
        for s in range(ns):
            gaps = []
            for k1, k2 in combinations(range(nk), 2):
                vbs = evs[k1, s, :][occs[k1, s, :] >= 1e-4]
                cbs = evs[k2, s, :][occs[k2, s, :] < 1e-4]
                cb = np.min(cbs)
                vb = np.max(vbs)
                gap = cb - vb
                gaps.append((k1, k2, gap, vb, cb))
            gaps = np.array(gaps)
            index = np.argmin(gaps[:, 2])
            if gaps[index, 2] < 0.1:
                # probably metallic along this spin channel
                results.append(None)
            if gaps[index, 2] >= 0.1:
                k1, k2, value, vbm, cbm = gaps[index]
                k1, k2 = int(k1), int(k2)
                kp1 = np.dot(kpts[k1], self.atoms.cell.T) / (2 * np.pi)
                kp2 = np.dot(kpts[k2], self.atoms.cell.T) / (2 * np.pi)
                kc1 = kcoords[k1]
                kc2 = kcoords[k2]
                indirect = IndirectBandGap(
                    value=value,
                    spin_index=s,
                    k_index1=k1,
                    k_index2=k2,
                    kpoint1=kp1,
                    kpoint2=kp2,
                    k_axis_coords1=kc1,
                    k_axis_coords2=kc2,
                    vbm=vbm,
                    cbm=cbm,
                )
                results.append(indirect)
        return results

    def get_indirect_gap(self, spin):
        """ Returns indirect band gap for given spin channel or None if metallic along this spin channel."""
        results = self._find_indirect_gap()
        s = self._spin2index(spin)
        return results[s]

    @property
    def bandgap(self):
        """Returns the fundamental band gap of the system."""
        ns = self.eigenvalues.shape[1]
        gaps = []
        for s in range(ns):
            dg = self.get_direct_gap(s)
            ig = self.get_indirect_gap(s)
            gaps.append(dg)
            gaps.append(ig)
        if any(l == None for l in gaps):
            return 0.0
        else:
            vals = [g.value for g in gaps]
            minvalue = min(vals)
            return gaps[vals.index(minvalue)]

    def __repr__(self):
        return "{}(bandpath={}, reference={}), band_gap={}".format(
            self.__class__.__name__, self.bandpath, self.reference, self.bandgap
        )

    def _spin2index(self, spin):
        if spin in [None, "none", "down", "dn", 0]:
            spin = 0
        elif spin in ["up", "UP", 1, 2]:
            spin = 1
        else:
            raise Exception("Spin channel not recognized.")
        return spin

    @property
    def atoms(self):
        return self._atoms

    @property
    def kpoints(self):
        return self._kpoints

    @property
    def kpoint_axis(self):
        return self._kpoint_axis

    @property
    def eigenvalues(self):
        return self._eigenvalues

    @property
    def occupations(self):
        return self._occupations

    @property
    def label_coords(self):
        return self._label_coords

    @property
    def kpoint_labels(self):
        return self._kpoint_labels

    @property
    def jumps(self):
        return self._jumps

    @property
    def fermi_level(self):
        return self._fermi_level

    @property
    def reference(self):
        return self._reference

    @property
    def band_extrema(self):
        return self._band_extrema

    @property
    def shift(self):
        return self._shift

    @property
    def bandpath(self):
        return self._bandpath

    def print_bandgap_information(self, spin="none"):
        spin = self._spin2index(spin)
        if self.bandgap == 0.0:
            logger.info("The system appears metallic.")
        else:
            try:
                dgap = self.get_direct_gap(spin)
            except:
                raise Exception("Could not determine direct band gap.")
            try:
                igap = self.get_indirect_gap(spin)
            except:
                raise Exception("Could not determine indirect band gap.")
            if dgap.value <= igap.value:
                # fundamental = "direct"
                logger.info(
                    f"From the spectrum, the fundamental band gap is {dgap.value} eV large and direct."
                )
                logger.info(
                    "The VBM and CBM are located at k = ( {:.4f} {:.4f} {:.4f} ) in units of the reciprocal lattice.".format(
                        *dgap.kpoint
                    )
                )
            else:
                # fundamental = "indirect"
                logger.info(
                    f"From the spectrum, the fundamental band gap is {igap.value} eV large and indirect."
                )
                logger.info(
                    "The VBM is located at k = ( {:.4f} {:.4f} {:.4f} ) in units of the reciprocal lattice.".format(
                        *igap.kpoint1
                    )
                )
                logger.info(
                    "The CBM is located at k = ( {:.4f} {:.4f} {:.4f} ) in units of the reciprocal lattice.".format(
                        *igap.kpoint2
                    )
                )
                logger.info(
                    "The smallest direct band gap is {:.4f} eV large and is located at k = ( {:.4f} {:.4f} {:.4f} ) in units of the reciprocal lattice.".format(
                        dgap.value, *dgap.kpoint
                    )
                )


class BandStructurePlot:
    """Context to draw band structure plot. Handles the correct shifting, labeling and axes limits."""

    def __init__(self, main=True, **kwargs) -> None:
        self.ax = kwargs.get("ax", None)
        assert (
            type(self.ax) != list
        ), "Axes object must be a single matplotlib.axes.Axes, not list."

        self.spectrum = kwargs.get("spectrum", None)
        self.spin = kwargs.get("spin", 0)
        self.set_data_from_spectrum()

        self.show_fermi_level = kwargs.get("show_fermi_level", True)
        self.fermi_level_color = kwargs.get("fermi_level_color", fermi_color)
        self.fermi_level_alpha = kwargs.get("fermi_level_alpha", 1.0)
        self.fermi_level_linestyle = kwargs.get("fermi_level_linestyle", "--")
        self.fermi_level_linewidth = kwargs.get(
            "fermi_level_linewidth", plt.rcParams["lines.linewidth"]
        )

        self.show_grid_lines = kwargs.get("show_grid_lines", True)
        self.grid_lines_axes = kwargs.get("show_grid_lines_axes", "x")
        self.grid_linestyle = kwargs.get("grid_linestyle", (0, (1, 1)))
        self.grid_linewidth = kwargs.get("grid_linewidth", 1.0)
        self.grid_linecolor = kwargs.get("grid_linecolor", mutedblack)

        self.show_jumps = kwargs.get("show_jumps", True)
        self.jumps_linewidth = kwargs.get(
            "jumps_linewidth", plt.rcParams["lines.linewidth"]
        )
        self.jumps_linestyle = kwargs.get("jumps_linestyle", "-")
        self.jumps_linecolor = kwargs.get("jumps_linecolor", mutedblack)

        self.show_bandstructure = kwargs.get("show_bandstructure", True)
        self.bands_color = kwargs.get("bands_color", mutedblack)
        self.bands_color = kwargs.get("color", mutedblack)
        self.bands_linewidth = kwargs.get(
            "bands_linewidth", plt.rcParams["lines.linewidth"]
        )
        self.bands_linewidth = kwargs.get("linewidth", plt.rcParams["lines.linewidth"])
        self.bands_linestyle = kwargs.get("bands_linestyle", "-")
        self.bands_linestyle = kwargs.get("linestyle", "-")
        self.bands_alpha = kwargs.get("bands_alpha", 1.0)
        self.bands_alpha = kwargs.get("alpha", 1.0)

        self.show_bandgap_vertices = kwargs.get("show_bandgap_vertices", True)

        self.window = kwargs.get("window", 3)
        self.y_tick_locator = kwargs.get("y_tick_locator", "auto")
        self.set_xy_axes_labels()
        self.set_xy_limits()
        self.set_energy_tick_locator()
        self.set_kpoint_labels()

        self.main = main

    def set_data_from_spectrum(self):
        spectrum = self.spectrum
        self.labels = spectrum.kpoint_labels.copy()
        self.labelcoords = spectrum.label_coords.copy()
        self.jumps = spectrum.jumps.copy()
        self.reference = spectrum.reference
        self.fermi_level = spectrum.fermi_level
        self.shift = spectrum.shift
        self.x = spectrum.kpoint_axis.copy()
        self.y = spectrum.eigenvalues[:, self.spin, :].copy() + self.shift
        self.fermi_level = spectrum.fermi_level
        self.band_extrema = spectrum.band_extrema

    def draw(self):
        ylocs = ticker.MultipleLocator(base=self.y_tick_locator)
        self.ax.yaxis.set_major_locator(ylocs)
        self.ax.set_xlabel(self.xlabel, fontsize=plt.rcParams["axes.labelsize"])
        self.ax.set_ylabel(self.ylabel, fontsize=plt.rcParams["axes.labelsize"])
        self.ax.set_xlim(self.xlimits)
        self.ax.set_ylim(self.ylimits)
        self.ax.set_xticks(self.xlabelcoords)
        self.ax.set_xticklabels(self.xlabels, fontsize=plt.rcParams["axes.labelsize"])
        self.ax.tick_params(axis="x", which="both", length=0)
        if self.show_grid_lines and self.main:
            self.ax.grid(
                b=self.show_grid_lines,
                which="major",
                axis=self.grid_lines_axes,
                linestyle=self.grid_linestyle,
                linewidth=self.grid_linewidth,
                color=self.grid_linecolor,
            )
        if self.show_jumps and self.main:
            for j in self.jumps:
                self.ax.axvline(
                    x=j,
                    linestyle=self.jumps_linestyle,
                    color=self.jumps_linecolor,
                    linewidth=self.jumps_linewidth,
                )
        if self.show_fermi_level and self.main:
            self._show_fermi_level()
        if self.show_bandgap_vertices and self.main:
            self._show_bandgap_vertices()
        if self.show_bandstructure and self.main:
            self.ax.plot(
                self.x,
                self.y,
                color=self.bands_color,
                alpha=self.bands_alpha,
                linewidth=self.bands_linewidth,
                linestyle=self.bands_linestyle,
            )

    def set_xy_axes_labels(self):
        if self.reference in ["fermi level", "VBM", "middle"]:
            ylabel = r"E - E$_{\mathrm{F}}$ [eV]"
        elif self.reference == "vacuum":
            ylabel = r"E - E$_{\mathrm{vacuum}}$ [eV]"
        else:
            ylabel = r"E [eV]"
        xlabel = ""
        self.xlabel = xlabel
        self.ylabel = ylabel

    def set_xy_limits(self):
        window = self.window
        x, y = self.x, self.y
        if isinstance(window, (float, int)):
            lower_ylimit, upper_ylimit = (-window, window)
            if self.reference in ["work function", "user-specified", "vacuum"]:
                lower_ylimit, upper_ylimit = (-window + self.shift, window + self.shift)
        elif len(window) == 2:
            lower_ylimit, upper_ylimit = window[0], window[1]
            if self.reference in ["work function", "user-specified", "vacuum"]:
                lower_ylimit, upper_ylimit = (window[0], window[1])
        else:
            logger.error("Energy window not recognized.")
            lower_ylimit, upper_ylimit = self.ax.get_ylim()
        lower_xlimit = 0.0
        upper_xlimit = np.max(x)
        self.xlimits = (lower_xlimit, upper_xlimit)
        self.ylimits = (lower_ylimit, upper_ylimit)

    def set_energy_tick_locator(self):
        if self.y_tick_locator == "auto":
            a, b = self.ylimits
            if (b - a) < 6:
                self.y_tick_locator = 0.5
            elif (b - a) < 9:
                self.y_tick_locator = 1
            else:
                self.y_tick_locator = 2
        else:
            assert isinstance(
                self.y_tick_locator, (int, float)
            ), "DOS tick locator must be int or float."

    def set_kpoint_labels(self):
        def pretty(kpt):
            if kpt == "G":
                kpt = r"$\Gamma$"
            elif len(kpt) == 2:
                kpt = kpt[0] + "$_" + kpt[1] + "$"
            return kpt

        labels = self.labels
        labels = [pretty(j) for j in labels]
        coords = self.labelcoords
        i = 1
        while i < len(labels):
            if coords[i - 1] == coords[i]:
                labels[i - 1] = labels[i - 1] + "|" + labels[i]
                labels.pop(i)
                coords.pop(i)
            else:
                i += 1

        self.xlabels = labels
        self.xlabelcoords = coords

    def _show_fermi_level(self):
        reference = self.spectrum.reference
        value = self.spectrum.shift
        if reference in ["user-specified"]:
            mark = value
        elif reference in ["vacuum"]:
            mark = value + (self.band_extrema[0] - self.fermi_level)
        else:
            mark = 0.00

        self.ax.axhline(
            y=mark,
            color=self.fermi_level_color,
            alpha=self.fermi_level_alpha,
            linestyle=self.fermi_level_linestyle,
            linewidth=self.fermi_level_linewidth,
        )

    def _show_bandgap_vertices(self):
        vertices = self._get_gap_vertices()
        i = 0
        colors = ["#393B79", "#3182BD"]
        for v in vertices:
            self.ax.plot(
                v[0],
                v[1],
                color=colors[i],
                linestyle=(0, (1, 1)),
                linewidth=plt.rcParams["lines.linewidth"],
                alpha=0.8,
            )
            self.ax.scatter(v[0], v[1], c=colors[i], alpha=0.8)
            i += 1

    def _get_gap_vertices(self):
        vertices = []
        indirect_gap = self.spectrum.get_indirect_gap(self.spin)
        direct_gap = self.spectrum.get_direct_gap(self.spin)
        mark_i, mark_d = False, False
        x1, x2, y1, y2 = None, None, None, None
        if indirect_gap != None:
            mark_i = True
        if direct_gap != None:
            mark_d = True
        if mark_i:
            x1, x2 = indirect_gap.k_axis_coords1, indirect_gap.k_axis_coords2
            y1, y2 = indirect_gap.vbm + self.shift, indirect_gap.cbm + self.shift
            vertices.append([(x1, x2), (y1, y2)])
        if mark_d:
            x1, x2 = direct_gap.k_axis_coords, direct_gap.k_axis_coords
            y1 = direct_gap.vbm + self.shift
            y2 = direct_gap.cbm + self.shift
            vertices.append([(x1, x2), (y1, y2)])
        return vertices


class MullikenBandStructurePlot(BandStructurePlot):
    """Context to draw mulliken band structures. Handles legends, color maps, etc."""

    def __init__(self, contributions=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.contributions = contributions

        self.mode = kwargs.get("mode", "lines")
        self.capstyle = kwargs.get("capstyle", "round")

        self.interpolate = kwargs.get("interpolate", False)
        self.interpolation_step = kwargs.get("interpolation_step", 0.01)

        self.norm = kwargs.get("norm", Normalize(vmin=0.0, vmax=1.0))

        self.scale_width = kwargs.get("scale_width", True)
        self.scale_width_factor = kwargs.get("scale_width_factor", 2)

        self.colors = kwargs.get("colors", ["red", "blue", "green"])
        self.cmaps = [self._color_to_alpha_cmap(c) for c in self.colors]

        self.labels = kwargs.get("labels", [1, 2, 3])

        self.show_legend = kwargs.get("show_legend", True)
        self.legend_linewidth = kwargs.get(
            "legend_linewidth", plt.rcParams["lines.linewidth"]
        )
        self.legend_frameon = kwargs.get(
            "legend_frameon", plt.rcParams["legend.frameon"]
        )
        self.legend_fancybox = kwargs.get(
            "legend_fancybox", plt.rcParams["legend.fancybox"]
        )
        self.legend_borderpad = kwargs.get(
            "legend_borderpad", plt.rcParams["legend.borderpad"]
        )
        self.legend_loc = kwargs.get("legend_loc", "upper right")
        self.legend_handlelength = kwargs.get(
            "legend_handlelength", plt.rcParams["legend.handlelength"]
        )

        self.show_colorbar = kwargs.get("show_colorbar", False)

    def draw(self):
        ylocs = ticker.MultipleLocator(base=self.y_tick_locator)
        self.ax.yaxis.set_major_locator(ylocs)
        self.ax.set_xlabel(self.xlabel, fontsize=plt.rcParams["axes.labelsize"])
        self.ax.set_ylabel(self.ylabel, fontsize=plt.rcParams["axes.labelsize"])
        self.ax.set_xlim(self.xlimits)
        self.ax.set_ylim(self.ylimits)
        self.ax.set_xticks(self.xlabelcoords)
        self.ax.set_xticklabels(self.xlabels, fontsize=plt.rcParams["axes.labelsize"])
        self.ax.tick_params(axis="x", which="both", length=0)
        if self.show_grid_lines and self.main:
            self.ax.grid(
                b=self.show_grid_lines,
                which="major",
                axis=self.grid_lines_axes,
                linestyle=self.grid_linestyle,
                linewidth=self.grid_linewidth,
                color=mutedblack,
            )
        if self.show_jumps and self.main:
            for j in self.jumps:
                self.ax.axvline(
                    x=j,
                    linestyle=self.jumps_linestyle,
                    color=self.jumps_linecolor,
                    linewidth=self.jumps_linewidth,
                )
        if self.show_fermi_level and self.main:
            self._show_fermi_level()
        if self.show_bandgap_vertices and self.main:
            self._show_bandgap_vertices()
        if self.show_bandstructure and self.main:
            self.ax.plot(
                self.x,
                self.y,
                color=self.bands_color,
                alpha=self.bands_alpha,
                linewidth=self.bands_linewidth,
                linestyle=self.bands_linestyle,
            )

        for band in range(self.y.shape[1]):
            band_x = self.x.copy()
            band_y = self.y[:, band].copy()
            if self.mode == "majority":
                con = self._get_majority_contribution()
                band_width = con.contribution[:, self.spin, band].copy()
                self.plot_linecollection(band_x, band_y, band_width, self.cmaps)
            elif self.mode == "lines":
                for i, con in enumerate(self.contributions):
                    band_width = con.contribution[:, self.spin, band].copy()
                    self.plot_linecollection(band_x, band_y, band_width, self.cmaps[i])
            elif self.mode == "scatter":
                for i, con in enumerate(self.contributions):
                    band_width = con.contribution[:, self.spin, band].copy()
                    self.plot_scatter(band_x, band_y, band_width, self.cmaps[i])
            elif self.mode == "gradient":
                con = self._get_difference_contribution()
                band_width = con.contribution[:, self.spin, band].copy()
                self.plot_linecollection(band_x, band_y, band_width, self.cmaps)
            else:
                raise Exception(f"Mode {self.mode} not implemented.")

        if self.show_colorbar:
            self._show_colorbar()
        if self.show_legend:
            self._show_legend()

    def plot_linecollection(self, band_x, band_y, band_width, cmap):
        band_x = band_x.copy()
        band_y = band_y.copy()
        band_width = band_width.copy()
        if self.interpolate:
            band_x, band_y, band_width = self.interpolate_bands_1d(
                band_x, band_y, band_width, self.interpolation_step
            )
        band_width = band_width[:-1]
        points = np.array([band_x, band_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate(
            [points[:-1], points[1:]], axis=1
        )  # this reshapes it into (x1, x2) (y1, y2) pairs
        if self.scale_width:
            lwidths = band_width.copy() * self.scale_width_factor
        else:
            lwidths = self.bands_linewidth
        lc = LineCollection(
            segments,
            linewidths=lwidths,
            cmap=cmap,
            norm=self.norm,
            capstyle=self.capstyle,
        )
        lc.set_array(band_width)
        self.ax.add_collection(lc)

    def plot_scatter(self, band_x, band_y, band_width, cmap):
        band_x = band_x.copy()
        band_y = band_y.copy()
        band_width = band_width.copy()
        if self.interpolate:
            band_x, band_y, band_width = self.interpolate_bands_1d(
                band_x, band_y, band_width, self.interpolation_step
            )
        if self.scale_width:
            swidths = band_width.copy() * self.scale_width_factor
        else:
            swidths = 1.5
        self.ax.scatter(band_x, band_y, c=swidths, cmap=cmap, norm=self.norm, s=swidths)

    def interpolate_bands_1d(self, band_x, band_y, band_width, interpolation_step):
        f1 = interpolate.interp1d(band_x, band_y)
        f2 = interpolate.interp1d(band_x, band_width)
        band_x = np.arange(0, np.max(band_x), interpolation_step)
        band_y = f1(band_x)
        band_width = f2(band_x)
        return band_x, band_y, band_width

    def _color_to_alpha_cmap(self, color):
        cmap = LinearSegmentedColormap.from_list("", ["white", color])
        my_cmap = cmap(np.arange(cmap.N))
        my_cmap[:, -1] = np.linspace(0, 1, cmap.N)  # this adds alpha
        my_cmap = ListedColormap(my_cmap)
        return my_cmap

    def _show_colorbar(self):
        clb = plt.colorbar(
            plt.cm.ScalarMappable(norm=self.norm, cmap=self.cmaps), ax=self.ax
        )
        if self.mode == "majority":
            clb.set_ticks(range(1, len(self.labels) + 1))
        elif self.mode == "gradient":
            clb.set_ticks([-1, 1])
        clb.set_ticklabels(self.labels)

    def _get_majority_contribution(self):
        assert (
            len(self.contributions) > 1
        ), "Majority projection only makes sense for more than one contribution."
        con = np.zeros(self.contributions[0].contribution.shape)
        for i, s, j in np.ndindex(con.shape):
            # at each kpoint i, each spin s, each state j, compare which value is largest
            l = [c.contribution[i, s, j] for c in self.contributions]
            k = l.index(max(l))
            # the index of the largest value is assigned to this point
            con[i, s, j] = k + 1
        fake_con = namedtuple("fake_con", ["Uhh", "contribution", "eeeeh"])
        contributions = fake_con("Uh?", con, "eeeh")
        self.scale_width = False
        self.cmaps = ListedColormap(self.colors)
        self.norm = BoundaryNorm(
            [0.5 + j for j in range(len(self.colors))] + [len(self.colors) + 0.5],
            self.cmaps.N,
        )
        return contributions

    def _show_legend(self):
        handles = []
        for c, l in zip(self.colors, self.labels):
            handles.append(Line2D([0], [0], color=c, label=l, lw=self.legend_linewidth))
        self.ax.legend(
            handles=handles,
            frameon=self.legend_frameon,
            fancybox=self.legend_fancybox,
            borderpad=self.legend_borderpad,
            loc=self.legend_loc,
            handlelength=self.legend_handlelength,
            fontsize=plt.rcParams["legend.fontsize"],
        )

    def _get_difference_contribution(self):

        assert (
            len(self.contributions) == 2
        ), "Difference contribution is only possible for exactly two contributions."
        con = self.contributions[1] - self.contributions[0]
        cmap = LinearSegmentedColormap.from_list("", self.colors)
        my_cmap = cmap(np.arange(cmap.N))
        my_cmap = ListedColormap(my_cmap)
        self.norm = Normalize(vmin=-1.0, vmax=1.0)
        self.cmaps = cmap
        return con
