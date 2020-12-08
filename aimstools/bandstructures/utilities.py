from aimstools.misc import *

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

import numpy as np


class BandStructurePlot:
    def __init__(
        self,
        spectrum=None,
        spin=None,
        ref=None,
        shift=None,
        window=None,
        vbm=None,
        cbm=None,
        direct_gap=None,
        indirect_gap=None,
        fermi_level=None,
        mark_fermi_level=None,
        mark_gap=False,
        main=False,
    ) -> None:
        self.ax = plt.gca()
        self.spectrum = spectrum
        x = spectrum.kpoint_axis.copy()
        y = spectrum.eigenvalues[:, spin, :].copy() - shift
        self.xy = (x, y)
        self.ref = ref
        self.shift = shift
        self.fermi_level = fermi_level
        self.vbm, self.cbm = self.set_vbm_cbm(vbm, cbm)
        self.direct_gap, self.indirect_gap = direct_gap, indirect_gap
        self.window = window
        self.labels = spectrum.kpoint_labels.copy()
        self.labelcoords = spectrum.label_coords.copy()
        self.jumps = spectrum.jumps.copy()
        self.xlabel, self.ylabel = self.set_xy_labels()
        self.xlimits, self.ylimits = self.set_xy_limits()
        self.mark_fermi_level = mark_fermi_level
        self.mark_gap = mark_gap
        self.main = main

    def draw(self):
        ylocs = ticker.MultipleLocator(base=0.5)
        self.ax.yaxis.set_major_locator(ylocs)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_xlim(self.xlimits)
        self.ax.set_ylim(self.ylimits)
        labels, coords = self.set_x_labels()
        self.ax.set_xticks(coords)
        self.ax.set_xticklabels(labels)
        self.ax.tick_params(axis=u"x", which=u"both", length=0)
        if self.main:
            self.ax.grid(
                b=True, which="major", axis="x", linestyle=(0, (1, 1)), linewidth=1.0
            )
            for j in self.jumps:
                self.ax.axvline(
                    x=j,
                    linestyle="-",
                    color=darkgray,
                    linewidth=mpllinewidth,
                )
            if self.mark_fermi_level not in ["None", "none", None, False]:
                self.show_fermi_level()
            if self.mark_gap not in ["None", "none", None, False]:
                self.show_vertices()
        return self.ax

    def set_vbm_cbm(self, vbm, cbm):
        if self.ref not in ["work function", "user-specified"]:
            vbm -= self.fermi_level
            cbm -= self.fermi_level
        else:
            vbm -= self.shift
            cbm -= self.shift
        return vbm, cbm

    def set_xy_labels(self):
        if self.ref in ["fermi level", "VBM", "middle"]:
            ylabel = r"E - E$_{\mathrm{F}}$ [eV]"
        elif self.ref == "work function":
            ylabel = r"E - E$_{vacuum}$ [eV]"
        else:
            ylabel = r"E [eV]"
        xlabel = ""
        return (xlabel, ylabel)

    def set_xy_limits(self):
        window = self.window
        x, y = self.xy
        if (type(window) == float) or (type(window) == int):
            lower_ylimit, upper_ylimit = (-window, window)
            if self.ref in ["work function", "user-specified"]:
                lower_ylimit, upper_ylimit = (-window - self.shift, window - self.shift)
        elif len(window) == 2:
            lower_ylimit, upper_ylimit = window[0], window[1]
            if self.ref in ["work function", "user-specified"]:
                lower_ylimit, upper_ylimit = (
                    window[0],
                    window[1],
                )
        else:
            logger.error("Energy window not recognized.")
            lower_ylimit, upper_ylimit = self.ax.get_ylim()
        lower_xlimit = 0.0
        upper_xlimit = np.max(x)
        return [(lower_xlimit, upper_xlimit), (lower_ylimit, upper_ylimit)]

    def set_x_labels(self):
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

        return labels, coords

    def show_fermi_level(self):
        ref = self.ref
        value = self.shift
        color = self.mark_fermi_level
        if ref in ["work function", "user-specified"]:
            mark = -value
        else:
            mark = 0.00

        self.ax.axhline(
            y=mark,
            color=color,
            alpha=fermi_alpha,
            linestyle="--",
            linewidth=mpllinewidth,
        )

    def show_vertices(self):
        vertices = self.get_gap_vertices()
        i = 0
        colors = ["#393B79", "#3182BD"]
        for v in vertices:
            self.ax.plot(
                v[0],
                v[1],
                color=colors[i],
                linestyle=(0, (1, 1)),
                linewidth=mpllinewidth,
                alpha=0.8,
            )
            self.ax.scatter(v[0], v[1], c=colors[i], alpha=0.8)
            i += 1

    def get_gap_vertices(self):
        vertices = []
        vbm = self.vbm
        cbm = self.cbm
        shift = self.shift
        indirect_gap = self.indirect_gap
        direct_gap = self.direct_gap
        mark_i, mark_d = False, False
        x1, x2, y1, y2 = None, None, None, None
        if indirect_gap != None:
            mark_i = True
        if direct_gap != None:
            mark_d = True
        if mark_i:
            x1, x2 = indirect_gap.axis_coord1, indirect_gap.axis_coord2
            y1, y2 = indirect_gap.e1 - shift, indirect_gap.e2 - shift
            vertices.append([(x1, x2), (y1, y2)])
        if mark_d:
            x1, x2 = direct_gap.axis_coord, direct_gap.axis_coord
            y1 = direct_gap.e1 - shift
            y2 = direct_gap.e2 - shift
            vertices.append([(x1, x2), (y1, y2)])
        return vertices


class MullikenBandStructurePlot:
    def __init__(
        self,
        x=None,
        y=None,
        spin=None,
        con=None,
        cmap=None,
        mode="lines",
        interpolate=False,
        norm=None,
        scale_width=2,
    ) -> None:
        self.ax = plt.gca()
        self.x = x.copy()
        self.y = y.copy()
        self.spin = spin
        self.con = con
        self.cmap = cmap
        self.mode = mode
        self.interpoalte = interpolate
        self.norm = norm or Normalize(vmin=0.0, vmax=1.0)
        self.scale_width = scale_width

    def draw(self):
        x = self.x
        y = self.y
        for band in range(y.shape[1]):
            band_x = x.copy()
            band_y = y[:, band].copy()
            band_width = self.con.contribution[:, self.spin, band].copy()
            if self.mode == "lines":
                self.plot_linecollection(band_x, band_y, band_width)
            elif self.mode == "scatter":
                self.plot_scatter(band_x, band_y, band_width)
        return self.ax

    def plot_linecollection(self, band_x, band_y, band_width):
        axes = self.ax
        band_width = band_width[:-1]
        points = np.array([band_x, band_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate(
            [points[:-1], points[1:]], axis=1
        )  # this reshapes it into (x1, x2) (y1, y2) pairs
        if self.scale_width in [False, None, "none", 0]:
            lwidths = 1
        else:
            lwidths = band_width.copy() * self.scale_width
        lc = LineCollection(
            segments,
            linewidths=lwidths,
            cmap=self.cmap,
            norm=self.norm,
            capstyle="round",
        )
        lc.set_array(band_width)
        axes.add_collection(lc)

    def plot_scatter(self, band_x, band_y, band_width):
        axes = self.ax
        if self.scale_width in [False, None, "none", 0]:
            swidths = 1
        else:
            swidths = band_width.copy() * self.scale_width
        axes.scatter(
            band_x,
            band_y,
            c=swidths,
            cmap=self.cmap,
            norm=self.norm,
            s=(band_width * 2),
        )

    def interpol(self):
        # if interpolation_step != False:
        #     f1 = interpolate.interp1d(x, band_y)
        #     f2 = interpolate.interp1d(x, band_width)
        #     band_x = np.arange(0, np.max(x), self.interpolation_step)
        #     band_y = f1(band_x)
        #     band_width = f2(band_x)
        pass
