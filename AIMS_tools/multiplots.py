import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from AIMS_tools.misc import *
from AIMS_tools import bandstructure, dos


def combine(nrows=1, ncols=1, list_of_axes=[], list_of_ratios=[]):
    """ Combines an arbitrary number of band structures or densities of states.

    Automatically detects whether band structures contain SOC information. If yes, ZORA and SOC are overlaid.

    Example:
        >>> from AIMS_tools import multiplots, bandstructure, dos
        >>> import matplotlib.pyplot as plt
        >>> bs = bandstructure.bandstructure("directory")
        >>> ds = dos.density_of_states("directory")
        >>> combi = multiplots.combine(nrows=1, ncols=2, list_of_axes=[bs, ds], list_of_ratios=[3,1])
        >>> plt.show()
        >>> plt.savefig("Name.png", dpi=300, transparent=False, bbox_inches="tight", facecolor="white")
  
    Args:
        nrows (int): Number of rows.
        ncols (int): Number of columns.
        list_of_axes (list): List of band structure or dos class objects to plot.
        list_of_ratios (list): List of width ratios. Must have same length as list_of_axes.
    
    Returns:
        figure: matplotlib figure object
    """
    if list_of_ratios == []:
        list_of_ratios = [1 for x in range(len(list_of_axes))]
    fig = plt.figure(constrained_layout=True, figsize=(nrows * 4, ncols * 3))
    spec = gridspec.GridSpec(
        ncols=ncols, nrows=nrows, figure=fig, width_ratios=list_of_ratios
    )

    indices = np.array(range(len(list_of_axes))).reshape((nrows, ncols))

    for ax in range(len(list_of_axes)):
        axes = fig.add_subplot(spec[ax])
        var = list_of_axes[ax]
        plt.sca(axes)
        if str(var) == "band structure":
            if var.active_SOC == True:
                axes = overlay_ZORA_SOC(var.path, axes=axes, fig=fig)
                if ax == 0:
                    ymin, ymax = axes.get_ylim()
                if ax != 0:
                    axes.set_ylim([ymin, ymax])
                    index = np.argwhere(indices == ax)
                    if index[0][1] != 0:
                        axes.set_ylabel("")
                        axes.set_yticks([])
            else:
                axes = var.plot(axes=axes, fig=fig)
                if ax == 0:
                    ymin, ymax = axes.get_ylim()
                if ax != 0:
                    axes.set_ylim([ymin, ymax])
                    index = np.argwhere(indices == ax)
                    if index[0][1] != 0:
                        axes.set_ylabel("")
                        axes.set_yticks([])
        if str(var) == "DOS":
            axes = var.plot_all_atomic_dos(fig=fig, axes=axes)
            if ax == 0:
                ymin, ymax = axes.get_ylim()
            if ax != 0:
                axes.set_ylim([ymin, ymax])
                xmax = []
                for line in axes.lines:
                    xmax.append(max(line.get_xdata()))
                axes.set_xlim(0, max(xmax) * 1.05)
                index = np.argwhere(indices == ax)
                if index[0][1] != 0:
                    axes.set_ylabel("")
                    axes.set_yticks([])
    return fig


def overlay_ZORA_SOC(
    BSpath,
    fig=None,
    axes=None,
    title="",
    ZORA_color="lightgray",
    SOC_color="crimson",
    var_energy_limits=1.0,
    fix_energy_limits=[],
    zorakwargs={"alpha": 1},
    sockwargs={"linestyle": "-"},
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


# def overlay_noGW_GW(
#     noGWpath,
#     GWpath,
#     fig=None,
#     axes=None,
#     title="",
#     noGW_color="black",
#     GW_color="royalblue",
#     var_energy_limits=1.0,
#     fix_energy_limits=[],
#     noGWkwargs={"alpha": 0.8},
#     GWkwargs={"linestyle": "--"},
# ):
#     """ Overlays a bandstructure plot with GW and without GW.

#     Example:
#         >>> from AIMS_tools import multiplots
#         >>> import matplotlib.pyplot as plt
#         >>> import numpy as np
#         >>> combi = multiplots.overlay_noGW_GW("noGW_outputfile", "GW_outputfile")
#         >>> plt.show()
#         >>> plt.savefig("Name.png", dpi=300, transparent=False, bbox_inches="tight", facecolor="white")

#     .. image:: ../pictures/MoSe2_GW+noGW.png
#         :width: 200px
#         :align: center
#         :height: 250px

#     Args:
#         noGWpath (str): Path to band structure calculation output file.
#         GWpath (str): Path to GW band structure calculation output file.
#         fig (matplotlib figure): Figure to draw the plot on.
#         axes (matplotlib axes): Axes to draw the plot on.
#         title (str, optional): Title of plot. Defaults to "".
#         noGW_color (str, optional): Color of band structure lines. Defaults to "gray".
#         GW_color (str, optional): Color of GW lines. Defaults to "royalblue".
#         var_energy_limits (float, optional): Variable energy range above and below the band gap to show. Defaults to 1.0.
#         fix_energy_limits (list, optional): List of lower and upper energy limits to show. Defaults to [].
#         **noGWkwargs (dict, optional): noGWkwargs are passed to the noGW.plot() function. Defaults to {"alpha":0.8}.
#         **GWkwargs (dict, optional): GWkwargs are passed to the GW.plot() function. Defaults to {"linestyle": "--"}.

#     Returns:
#         axes: matplotlib axes object"""
#     # noGW section
#     noGW = bandstructure.bandstructure(noGWpath, get_SOC=False)
#     with open(noGW.path.joinpath("control.in"), "r") as file:
#         for line in file.readlines():
#             if "xc" in line:
#                 xc = line.split()[-1]
#     if fig == None:
#         fig = plt.figure(figsize=(len(noGW.kpath) / 2, 3))
#     if axes != None:
#         axes = plt.gca()
#     else:
#         axes = plt.subplot2grid((1, 1), (0, 0), fig=fig)
#     noGW.plot(
#         fig=fig,
#         axes=axes,
#         color=noGW_color,
#         var_energy_limits=var_energy_limits,
#         fix_energy_limits=fix_energy_limits,
#         kwargs=noGWkwargs,
#     )
#     noGW_line = Line2D([0], [0], color=noGW_color, label=xc, lw=1.5)
#     # GW section
#     GW = bandstructure.bandstructure(GWpath, get_SOC=False)
#     GW.properties()
#     GW.plot(
#         fig=fig,
#         axes=axes,
#         color=GW_color,
#         var_energy_limits=var_energy_limits,
#         fix_energy_limits=fix_energy_limits,
#         kwargs=GWkwargs,
#     )
#     GW_line = Line2D([0], [0], color=GW_color, label="GW@{}".format(xc), lw=1.5)
#     axes.legend(
#         handles=[noGW_line, GW_line],
#         frameon=True,
#         fancybox=False,
#         borderpad=0.4,
#         ncol=2,
#         loc="upper right",
#     )
#     fig.suptitle(title)
#     return axes
