# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 09:12:57 2019

@author: roman
"""

import sys, os
import argparse
from pathlib import Path as Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from AIMS_tools.misc import *
from AIMS_tools import bandstructure, dos


def combine_bs_dos(BSpath, DOSpath, title="", fix_energy_limits=[]):
    """ Combines a band structure plot and densities of states plot.

    Automatically detects whether BSpath contains SOC information. If yes, overlay_ZORA_SOC() is invoked.

    Example:
        >>> from AIMS_tools import multiplots
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> combi = multiplots.combine_bs_dos("band_structure_outputfile", "dos_outputfile")
        >>> plt.show()
        >>> plt.savefig("Name.png", dpi=300, transparent=False, bbox_inches="tight", facecolor="white")

    .. image:: ../pictures/MoS2_BS+DOS.png
        :width: 250px
        :align: center
        :height: 250px
    
    Args:
        BSpath (str): Path to band structure calculation output file.
        DOSpath (str): Path to density of states calculation output file.
        title (str, optional): Ttile of the plot
        fix_energy_limits (list, optional): List of lower and upper energy limits to show. Defaults to [].
    
    Returns:
        figure: matplotlib figure object
    """
    fig = plt.figure(constrained_layout=True, figsize=(4, 4))
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig, width_ratios=[3, 1])
    ax1 = fig.add_subplot(spec[0])
    ax2 = fig.add_subplot(spec[1])

    ## Handle bandstructures
    plt.sca(ax1)
    ZORA = bandstructure.bandstructure(BSpath, get_SOC=False)
    if ZORA.active_SOC == True:
        SOC = bandstructure.bandstructure(BSpath, get_SOC=True)
        ax1 = overlay_ZORA_SOC(
            BSpath, axes=ax1, fig=fig, fix_energy_limits=fix_energy_limits
        )
    else:
        ax1 = ZORA.plot(BSpath, axes=ax1, fig=fig, fix_energy_limits=fix_energy_limits)
    ymin, ymax = ax1.get_ylim()

    ## Handle DOS
    plt.sca(ax2)
    DOS = dos.DOS(DOSpath)
    ax2 = DOS.plot_all_atomic_dos(fig=fig, axes=ax2, fix_energy_limits=[ymin, ymax])
    ax2.set_ylabel("")
    ax2.set_yticks([])
    xmin, xmax = ax2.get_xlim()
    ax2.set_xlim(xmin, xmax)

    fig.suptitle(title)
    return fig


def overlay_ZORA_SOC(
    BSpath,
    fig=None,
    axes=None,
    title="",
    ZORA_color="black",
    SOC_color="crimson",
    var_energy_limits=1.0,
    fix_energy_limits=[],
    zorakwargs={"alpha": 0.8},
    sockwargs={"linestyle": "--"},
):
    """ Overlays a bandstructure plot with ZORA and SOC.

    Example:
        >>> from AIMS_tools import multiplots
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> combi = multiplots.overlay_ZORA_SOC("outputfile")
        >>> plt.show()
        >>> plt.savefig("Name.png", dpi=300, transparent=False, bbox_inches="tight", facecolor="white")

    .. image:: ../pictures/MoS2_ZORA+SOC.png
        :width: 220px
        :align: center
        :height: 250px

    Args:
        BSpath (str): Path to band structure calculation output file.
        fig (matplotlib figure): Figure to draw the plot on.
        axes (matplotlib axes): Axes to draw the plot on.
        title (str, optional): Title of plot. Defaults to "".
        ZORA_color (str, optional): Color of ZORA lines. Defaults to "gray".
        SOC_color (str, optional): Color of SOC lines. Defaults to "crimson".
        var_energy_limits (float, optional): Variable energy range above and below the band gap to show. Defaults to 1.0.
        fix_energy_limits (list, optional): List of lower and upper energy limits to show. Defaults to [].
        **zorakwargs (dict, optional): zorakwargs are passed to the ZORA.plot() function. Defaults to {"alpha":0.8}.
        **sockwargs (dict, optional): sockwargs are passed to the SOC.plot() function. Defaults to {"linestyle": "--"}.

    Returns:
        axes: matplotlib axes object"""
    # ZORA section
    ZORA = bandstructure.bandstructure(BSpath, get_SOC=False)
    if fig == None:
        fig = plt.figure(figsize=(len(ZORA.kpath) / 2, 3))
    if axes != None:
        axes = plt.gca()
    else:
        axes = plt.subplot2grid((1, 1), (0, 0), fig=fig)
    ZORA.plot(
        fig=fig,
        axes=axes,
        color=ZORA_color,
        var_energy_limits=var_energy_limits,
        fix_energy_limits=fix_energy_limits,
        kwargs=zorakwargs,
    )
    ZORA_line = Line2D([0], [0], color=ZORA_color, label="ZORA", lw=1.5)
    # SOC section
    SOC = bandstructure.bandstructure(BSpath, get_SOC=True)
    SOC.properties()
    SOC.plot(
        fig=fig,
        axes=axes,
        color=SOC_color,
        var_energy_limits=var_energy_limits,
        fix_energy_limits=fix_energy_limits,
        kwargs=sockwargs,
    )
    SOC_line = Line2D([0], [0], color=SOC_color, label="ZORA+SOC", lw=1.5)
    axes.legend(
        handles=[ZORA_line, SOC_line],
        frameon=True,
        fancybox=False,
        borderpad=0.4,
        ncol=2,
        loc="upper right",
    )
    fig.suptitle(title)
    return axes


def overlay_noGW_GW(
    noGWpath,
    GWpath,
    fig=None,
    axes=None,
    title="",
    noGW_color="black",
    GW_color="royalblue",
    var_energy_limits=1.0,
    fix_energy_limits=[],
    noGWkwargs={"alpha": 0.8},
    GWkwargs={"linestyle": "--"},
):
    """ Overlays a bandstructure plot with GW and without GW.

    Example:
        >>> from AIMS_tools import multiplots
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> combi = multiplots.overlay_noGW_GW("noGW_outputfile", "GW_outputfile")
        >>> plt.show()
        >>> plt.savefig("Name.png", dpi=300, transparent=False, bbox_inches="tight", facecolor="white")
    
    .. image:: ../pictures/MoSe2_GW+noGW.png
        :width: 200px
        :align: center
        :height: 250px

    Args:
        noGWpath (str): Path to band structure calculation output file.
        GWpath (str): Path to GW band structure calculation output file.
        fig (matplotlib figure): Figure to draw the plot on.
        axes (matplotlib axes): Axes to draw the plot on.
        title (str, optional): Title of plot. Defaults to "".
        noGW_color (str, optional): Color of band structure lines. Defaults to "gray".
        GW_color (str, optional): Color of GW lines. Defaults to "royalblue".
        var_energy_limits (float, optional): Variable energy range above and below the band gap to show. Defaults to 1.0.
        fix_energy_limits (list, optional): List of lower and upper energy limits to show. Defaults to [].
        **noGWkwargs (dict, optional): noGWkwargs are passed to the noGW.plot() function. Defaults to {"alpha":0.8}.
        **GWkwargs (dict, optional): GWkwargs are passed to the GW.plot() function. Defaults to {"linestyle": "--"}.

    Returns:
        axes: matplotlib axes object"""
    # noGW section
    noGW = bandstructure.bandstructure(noGWpath, get_SOC=False)
    with open(noGW.path.joinpath("control.in"), "r") as file:
        for line in file.readlines():
            if "xc" in line:
                xc = line.split()[-1]
    if fig == None:
        fig = plt.figure(figsize=(len(noGW.kpath) / 2, 3))
    if axes != None:
        axes = plt.gca()
    else:
        axes = plt.subplot2grid((1, 1), (0, 0), fig=fig)
    noGW.plot(
        fig=fig,
        axes=axes,
        color=noGW_color,
        var_energy_limits=var_energy_limits,
        fix_energy_limits=fix_energy_limits,
        kwargs=noGWkwargs,
    )
    noGW_line = Line2D([0], [0], color=noGW_color, label=xc, lw=1.5)
    # GW section
    GW = bandstructure.bandstructure(GWpath, get_SOC=False)
    GW.properties()
    GW.plot(
        fig=fig,
        axes=axes,
        color=GW_color,
        var_energy_limits=var_energy_limits,
        fix_energy_limits=fix_energy_limits,
        kwargs=GWkwargs,
    )
    GW_line = Line2D([0], [0], color=GW_color, label="GW@{}".format(xc), lw=1.5)
    axes.legend(
        handles=[noGW_line, GW_line],
        frameon=True,
        fancybox=False,
        borderpad=0.4,
        ncol=2,
        loc="upper right",
    )
    fig.suptitle(title)
    return axes
