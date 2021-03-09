from aimstools.misc import *

from aimstools.bandstructures.base import BandStructureBaseClass
from aimstools.bandstructures.regular_bandstructure import RegularBandStructure
from aimstools.bandstructures.brillouinezone import BrillouineZone
from aimstools.bandstructures.mulliken_bandstructure import MullikenBandStructure

from matplotlib.lines import Line2D


class BandStructure:
    """Represents a collection of band structures.

    The bandstructure class is a wrapper for regular band structures and mulliken-projected band structures.
    To access these classes individually, see :class:`~aimstools.bandstructures.regular_bandstructure` and
    :class:`~aimstools.bandstructures.mulliken_bandstructure`. They are stored as properties (if calculated):

    >>> from aimstools import BandStructure
    >>> bs = BandStructure("/path/to/outputfile")
    >>> bs_zora = bs.bandstructure_zora
    >>> bs_soc = bs.bandstructure_soc
    >>> bs_mlk = bs.bandstructure_mulliken

    The :func:`~aimstools.bandstructures.bandstructure.plot` method creates a visualization out-of-the-box with some default settings
    and visualizes the Brillouine zone (show_BZ=True):

    >>> bs.plot(show_BZ=True)

    You can get an overview of band structure properties via:

    >>> bs.get_properties()

    Args:
        outputfile (str): Path to output file or output directory.
        mulliken_outputfile (str, optional): Path to output file or output directory for mulliken band structure, if different from band structure.

    """

    def __init__(self, outputfile, mulliken_outputfile=None) -> None:
        self.outputfile = outputfile
        self.mulliken_outputfile = mulliken_outputfile or outputfile
        self._bs = None
        self._bs_soc = None
        self._bs_mlk = None
        self._set_classes()

    def _set_classes(self):
        self.base = BandStructureBaseClass(outputfile=self.outputfile)
        self.soc = self.base.control["include_spin_orbit"]
        if "band structure" in self.base.tasks:
            self._bs = RegularBandStructure(outputfile=self.outputfile, soc=False)
            if self.soc:
                self._bs_soc = RegularBandStructure(
                    outputfile=self.outputfile, soc=True
                )
        if "mulliken-projected band structure" in self.base.tasks:
            self._bs_mlk = MullikenBandStructure(
                outputfile=self.mulliken_outputfile, soc=self.soc
            )

    def __repr__(self):
        return "{}(bandstructure_zora={}, bandstructure_soc={}, bandstructure_mulliken={})".format(
            self.__class__.__name__,
            self.bandstructure_zora,
            self.bandstructure_soc,
            self.bandstructure_mulliken,
        )

    @property
    def bandstructure_zora(self):
        """ Returns :class:`~aimstools.bandstructures.regular_bandstructure` without spin-orbit coupling."""
        return self._bs

    @property
    def bandstructure_soc(self):
        """ Returns :class:`~aimstools.bandstructures.regular_bandstructure` with spin-orbit coupling."""
        return self._bs_soc

    @property
    def bandstructure_mulliken(self):
        """ Returns :class:`~aimstools.bandstructures.mulliken_bandstructure`."""
        return self._bs_mlk

    def plot(
        self,
        show_BZ=True,
        scalar_bands_color=mutedblack,
        soc_bands_color="crimson",
        **kwargs
    ):
        """Utility function to quickly visualize band structure without much customization.

        Todo:
            Rewrite this part.
        """
        base = self.base
        if kwargs.get("spin") != None:
            raise Exception("Roman did not implement spin plotting in this class yet.")

        bp = kwargs.get("bandpath", None)
        _ = kwargs.pop("main", None)
        ncols = 0
        if "band structure" in base.tasks:
            ncols += 1
        if "mulliken-projected band structure" in base.tasks:
            ncols += 1
        if show_BZ:
            ncols += 1
        assert ncols > 0, "You need to specify a band structure."

        fig = plt.figure(figsize=(7 * ncols, 5))

        i = 0
        if "band structure" in base.tasks:
            i += 1
            ax_bs = fig.add_subplot(1, ncols, i)
            if self.soc == False:
                bs = self._bs
                ax_bs = bs.plot(axes=ax_bs, color=scalar_bands_color, **kwargs)
            if self.soc:
                bs = self._bs
                bs_soc = self._bs_soc
                ax_bs = bs.plot(
                    axes=ax_bs, color=scalar_bands_color, main=False, **kwargs
                )
                ax_bs = bs_soc.plot(
                    axes=ax_bs, color=soc_bands_color, main=True, **kwargs
                )
                handles = []
                handles.append(
                    Line2D([0], [0], color=scalar_bands_color, label="ZORA", lw=1.5)
                )
                handles.append(
                    Line2D([0], [0], color=soc_bands_color, label="ZORA+SOC", lw=1.5)
                )
                lgd = ax_bs.legend(
                    handles=handles,
                    frameon=True,
                    fancybox=False,
                    borderpad=0.4,
                    loc="upper right",
                )
                ax_bs.set_title("Band structure")
        if "mulliken-projected band structure" in base.tasks:
            i += 1
            ax_mlk = fig.add_subplot(1, ncols, i)
            bs = self._bs_mlk
            bs.plot_majority_contributions(axes=ax_mlk, **kwargs)
            ax_mlk.set_title("Majority contribution")
            if i == 2:
                ax_mlk.set_ylabel("")
                ax_mlk.set_yticks([])

        if show_BZ:
            i += 1
            if base.structure.is_2d():
                ax_bz = fig.add_subplot(1, ncols, i)
            else:
                ax_bz = fig.add_subplot(1, ncols, i, projection="3d")
            bp = bs.bandpath
            bz = BrillouineZone(
                base.structure, bp.path, special_points=bp.special_points
            )
            ax_bz = bz.plot(axes=ax_bz)
            ax_bz.set_title("Brillouine Zone")

        plt.show()

    def get_properties(self, bandstructureclass=None, spin="none"):
        """Utility function to print out band gap properties for given spin channel."""
        bs = bandstructureclass or (
            (self.bandstructure_soc or self.bandstructure_zora)
            or self.bandstructure_mulliken
        )

        soc = bs.soc
        bg = bs.bandgap.soc if soc else bs.bandgap.scalar
        logger.info(
            "The band structure has been calculated {} spin-orbit-coupling.".format(
                "with" if soc else "without"
            )
        )
        logger.info("The band gap from the output file is {:.4f} eV large.".format(bg))
        bs.spectrum.print_bandgap_information(spin=spin)
