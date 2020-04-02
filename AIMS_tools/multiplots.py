import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm
from matplotlib.lines import Line2D

from AIMS_tools.misc import *
from AIMS_tools import bandstructure, dos


def combine(
    nrows=1, ncols=1, results=[], ratios=[], titles=[], overlay_SOC=True, **kwargs
):
    """ Combines an arbitrary number of band structures or densities of states.

    Results must be a list of bandstructure or dos objects. If an element of results is a tuple of two bandstructure objects, both will be overlaid. If overlay_SOC is True and a bandstructure object has SOC enabled, ZORA and SOC will be overlaid.

    Example:
        >>> from AIMS_tools import multiplots, bandstructure, dos
        >>> import matplotlib.pyplot as plt
        >>> bs = bandstructure.bandstructure("directory")
        >>> ds = dos.density_of_states("directory")
        >>> combi = multiplots.combine(nrows=1, ncols=2, results=[bs, ds], ratios=[3,1])
        >>> plt.show()
        >>> plt.savefig("Name.png", dpi=300, transparent=False, bbox_inches="tight", facecolor="white")
  
    Args:
        nrows (int): Number of rows.
        ncols (int): Number of columns.
        results (list): List of band structure or dos class objects to plot.
        ratios (list): List of width ratios. Must have same length as results.
        titles (list): List of str containing subplot titles.        
    
    Returns:
        figure: matplotlib figure object
    """

    assert (nrows * ncols) <= len(
        results
    ), "Too many things to plot for number of rows and columns."

    if ratios == []:
        ratios = [1 for x in range(len(results))]
    fig = plt.figure(constrained_layout=True, figsize=(ncols * 6.4 / 2, nrows * 4.8),)
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig, width_ratios=ratios)

    indices = np.array(range(len(results))).reshape((nrows, ncols))

    for ax in range(len(results)):
        axes = fig.add_subplot(spec[ax])
        plt.sca(axes)
        var = results[ax]
        keywords = kwargs
        if type(var) != tuple:
            if str(var) == "band structure":
                if (var.active_SOC == True) and (
                    overlay_SOC in [True, "True", 1, "yes", "y"]
                ):
                    bs = bandstructure.bandstructure(var.path, get_SOC=False)
                    _, _ = (
                        keywords.pop("color", ""),
                        keywords.pop("var_energy_limits", []),
                    )
                    axes = overlay_bandstructures(
                        bandstrucs=[bs, var],
                        colors=["lightgray", "crimson"],
                        labels=["ZORA", "ZORA+SOC"],
                        axes=axes,
                        fig=fig,
                        var_energy_limits=2,
                        **keywords,
                    )
                    if ax == 0:
                        ymin, ymax = axes.get_ylim()
                    if ax != 0:
                        axes.set_ylim([ymin, ymax])
                        index = np.argwhere(indices == ax)
                        if index[0][1] != 0:
                            axes.set_ylabel("")
                            axes.set_yticks([])
                else:
                    axes = var.plot(axes=axes, fig=fig, **keywords)
                    if ax == 0:
                        ymin, ymax = axes.get_ylim()
                    if ax != 0:
                        axes.set_ylim([ymin, ymax])
                        index = np.argwhere(indices == ax)
                        if index[0][1] != 0:
                            axes.set_ylabel("")
                            axes.set_yticks([])
            if str(var) == "DOS":
                axes = var.plot_all_species(fig=fig, axes=axes, **keywords)
                if ax == 0:
                    ymin, ymax = axes.get_ylim()
                if ax != 0:
                    axes.set_ylim([ymin, ymax])
                    xmax = []
                    index = np.argwhere(indices == ax)
                    if index[0][1] != 0:
                        axes.set_ylabel("")
                        axes.set_yticks([])
                for line in axes.lines:
                    xmax.append(max(line.get_xdata()))
                axes.set_xlim(0, max(xmax) * 1.05)

            if titles != []:
                axes.set_title(titles[ax])

        elif type(var) == tuple:
            assert all(
                [str(v) == "band structure" for v in var]
            ), "Not all elements of tuple are band structures."
            overlay_bandstructures(bandstrucs=var, fig=fig, axes=axes, **keywords)
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


def overlay_bandstructures(
    bandstrucs=[], colors=[], labels=[], fig=None, axes=None, **kwargs
):
    """ Overlays an arbitrary number of band structures on the same axes.
 
    Args:
        results (list): List of band structure or dos class objects to plot.
        ratios (list): List of width ratios. Must have same length as results.
        titles (list): List of str containing subplot titles.
    
    Returns:
        figure: matplotlib figure object
    """

    cmap = matplotlib.cm.get_cmap("brg")
    if colors == []:
        colors = [cmap(i / len(bandstrucs)) for i in range(len(bandstrucs))]
    if labels == []:
        labels = [str(i) for i in range(len(bandstrucs))]

    nr1 = bandstrucs[0]
    kpath = nr1.kpath[0]
    for j in range(1, len(nr1.kpath)):
        kpath += "-{}".format(nr1.kpath[j])
    handles = []
    if fig == None:
        fig = plt.figure(figsize=(len(nr1.kpath) / 1.5, 3))
    if axes == None:
        axes = plt.gca()
    for ax in range(len(bandstrucs)):
        keywords = kwargs
        bs = bandstrucs[ax]
        if ax == 0:
            bs.plot(fig=fig, axes=axes, color=colors[ax], **keywords)
            handles.append(Line2D([0], [0], color=colors[ax], label=labels[ax], lw=1.5))
        else:
            bs.plot(
                fig=fig,
                axes=axes,
                color=colors[ax],
                fix_energy_limits=axes.get_ylim(),
                **keywords,
            )
            handles.append(Line2D([0], [0], color=colors[ax], label=labels[ax], lw=1.5))
    axes.legend(
        handles=handles,
        frameon=True,
        fancybox=False,
        borderpad=0.4,
        ncol=2,
        loc="upper right",
    )
    return axes
