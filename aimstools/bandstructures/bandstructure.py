from aimstools.misc import *

from aimstools.bandstructures.base import BandStructureBaseClass
from aimstools.bandstructures.regular_bandstructure import RegularBandStructure
from aimstools.bandstructures.brillouinezone import BrillouinZone
from aimstools.bandstructures.mulliken_bandstructure import MullikenBandStructure

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class BandStructure(BandStructureBaseClass):
    """Represents a collection of band structures.

    The bandstructure class is a wrapper for regular band structures and mulliken-projected band structures.
    To access these classes individually, see :class:`~aimstools.bandstructures.regular_bandstructure` and
    :class:`~aimstools.bandstructures.mulliken_bandstructure`. They are stored as properties (if calculated).

    >>> from aimstools import BandStructure
    >>> bs = BandStructure("/path/to/outputfile")

    The :func:`~aimstools.bandstructures.bandstructure.plot` is a wrapper that chooses a plotting method depending on the settings of the calculation. 

    >>> bs.plot()

    You can get an overview of band structure properties via:

    >>> bs.get_properties()

    Args:
        outputfile (str): Path to output file or output directory.

    """

    def __init__(self, outputfile) -> None:
        BandStructureBaseClass.__init__(self, outputfile=outputfile)
        self._regular_bandstructure_zora = None
        self._regular_bandstructure_soc = None
        self._mulliken_bandstructure_zora = None
        self._mulliken_bandstructure_soc = None
        if "band structure" in self.tasks:
            self._regular_bandstructure_zora = RegularBandStructure(
                outputfile=outputfile, soc=False
            )
            if self.control["include_spin_orbit"]:
                self._regular_bandstructure_soc = RegularBandStructure(
                    outputfile=outputfile, soc=True
                )
        if "mulliken-projected band structure" in self.tasks:
            # SOC and ZORA are mutually exclusive for MLK bands
            if not self.control["include_spin_orbit"]:
                self._mulliken_bandstructure_zora = MullikenBandStructure(
                    outputfile=outputfile, soc=False
                )
            else:
                self._mulliken_bandstructure_soc = MullikenBandStructure(
                    outputfile=outputfile, soc=True
                )

    def __repr__(self):
        return "{}(regular_bandstructure_zora={}, regular_bandstructure_soc={}, mulliken_bandstructure_zora={}, mulliken_bandstructure_soc={})".format(
            self.__class__.__name__,
            self._regular_bandstructure_zora,
            self._regular_bandstructure_soc,
            self._mulliken_bandstructure_zora,
            self._mulliken_bandstructure_soc,
        )

    @property
    def regular_bandstructure_zora(self):
        """ Returns :class:`~aimstools.bandstructures.regular_bandstructure` without spin-orbit coupling."""
        if self._regular_bandstructure_zora == None:
            raise Exception(
                "Regular band structure without spin-orbit coupling was not calculated."
            )
        return self._regular_bandstructure_zora

    @property
    def regular_bandstructure_soc(self):
        """ Returns :class:`~aimstools.bandstructures.regular_bandstructure` with spin-orbit coupling."""
        if self._regular_bandstructure_soc == None:
            raise Exception(
                "Regular band structure with spin-orbit coupling was not calculated."
            )
        return self._regular_bandstructure_soc

    @property
    def mulliken_bandstructure_zora(self):
        """ Returns :class:`~aimstools.bandstructures.mulliken_bandstructure` without spin-orbit coupling."""
        if self._mulliken_bandstructure_zora == None:
            raise Exception(
                "Mulliken-projected band structure without spin-orbit coupling was not calculated."
            )
        return self._mulliken_bandstructure_zora

    @property
    def mulliken_bandstructure_soc(self):
        """ Returns :class:`~aimstools.bandstructures.mulliken_bandstructure` with spin-orbit coupling."""
        if self._mulliken_bandstructure_soc == None:
            raise Exception(
                "Mulliken-projected band structure with spin-orbit coupling was not calculated."
            )
        return self._mulliken_bandstructure_soc

    def _choose_case(self):
        case, target = None, None
        if self._regular_bandstructure_zora is not None:
            case = "reg+zora"
        if self._regular_bandstructure_soc is not None:
            case = "reg+zora+soc"
        if self._mulliken_bandstructure_zora is not None:
            case = "mlk+zora"
        if self._mulliken_bandstructure_soc is not None:
            case = "mlk+zora+soc"

        assert (
            case is not None
        ), "Could not choose appropriate plotting case for this calculation setup."

        # regular cases
        if self.control.spin == "none" and not self.control.include_spin_orbit:
            target = "zora only"
        if self.control.spin == "collinear" and not self.control.include_spin_orbit:
            target = "two spin channels"
        if (
            self.control.spin == "none"
            and self.control.include_spin_orbit
            and "mlk" not in case
        ):
            target = "zora+soc"

        # mlk cases
        if (
            self.control.spin == "collinear"
            and self.control.include_spin_orbit
            and "mlk" not in case
        ):
            target = "soc only"
        if (
            self.control.spin == "collinear"
            and self.control.include_spin_orbit
            and "mlk" in case
        ):
            target = "soc only"
        if (
            self.control.spin == "none"
            and self.control.include_spin_orbit
            and "mlk" in case
        ):
            target = "soc only"

        assert (
            target is not None
        ), "Could not choose appropriate plotting target for this calculation setup."
        return (case, target)

    def _plot_both_spin_channels(self, axes=None, **kwargs):
        bs = self._regular_bandstructure_zora or self._mulliken_bandstructure_zora
        kwargs = kwargs.copy()
        kwargs.pop("reference", None)
        kwargs.pop("color", None)
        spin_up_color = kwargs.get("spin_up_color", "tab:blue")
        spin_down_color = kwargs.get("spin_down_color", "tab:cyan")
        axes = bs.plot(
            axes=axes,
            main=True,
            spin="up",
            color=spin_up_color,
            reference="fermi",
            **kwargs
        )
        axes = bs.plot(
            axes=axes,
            main=True,
            spin="down",
            color=spin_down_color,
            reference="fermi",
            **kwargs
        )
        handles = [
            Line2D(
                [0],
                [0],
                color=spin_up_color,
                label="up",
                linewidth=plt.rcParams["lines.linewidth"],
            ),
            Line2D(
                [0],
                [0],
                color=spin_down_color,
                label="down",
                linewidth=plt.rcParams["lines.linewidth"],
            ),
        ]
        lgd = axes.legend(
            handles=handles,
            frameon=plt.rcParams["legend.frameon"],
            fancybox=plt.rcParams["legend.fancybox"],
            borderpad=plt.rcParams["legend.borderpad"],
            handlelength=plt.rcParams["legend.handlelength"],
            loc="upper right",
        )
        return axes

    def _plot_zora_and_soc(self, axes=None, **kwargs):
        zora = self.regular_bandstructure_zora
        soc = self.regular_bandstructure_soc
        kwargs = kwargs.copy()
        kwargs.pop("color", None)
        zora_color = kwargs.get("zora_color", "tab:gray")
        soc_color = kwargs.get("soc_color", "tab:blue")
        axes = zora.plot(
            axes=axes, main=True, color=zora_color, linestyle="--", **kwargs
        )
        axes = soc.plot(axes=axes, main=True, color=soc_color, **kwargs)
        handles = [
            Line2D(
                [0],
                [0],
                color=zora_color,
                label="ZORA",
                linewidth=plt.rcParams["lines.linewidth"],
                linestyle="--",
            ),
            Line2D(
                [0],
                [0],
                color=soc_color,
                label="SOC",
                linewidth=plt.rcParams["lines.linewidth"],
            ),
        ]
        lgd = axes.legend(
            handles=handles,
            frameon=plt.rcParams["legend.frameon"],
            fancybox=plt.rcParams["legend.fancybox"],
            borderpad=plt.rcParams["legend.borderpad"],
            handlelength=plt.rcParams["legend.handlelength"],
            loc="upper right",
        )
        return axes

    def _plot_mulliken_projection(self, axes=None, **kwargs):
        if not self.control["include_spin_orbit"]:
            mlk = self.mulliken_bandstructure_zora
        else:
            mlk = self.mulliken_bandstructure_soc
        axes = mlk.plot_majority_contribution(axes=axes, **kwargs)
        return axes

    def plot_brillouin_zone(self, axes=None, bandpathstring=None, **kwargs):
        atoms = self.structure.atoms
        bz = BrillouinZone(atoms=atoms, bandpathstring=bandpathstring, **kwargs)
        bz.plot(axes=axes)
        return axes

    def plot(self, axes=None, **kwargs):
        if axes == None:
            ncols = 2
            nrows = 1
            projections = [["rectilinear", "3d"]]
            show_BZ = True
        else:
            ncols = 1
            nrows = 1
            projections = [["rectilinear"]]
            show_BZ = False

        case, target = self._choose_case()

        with AxesContext(
            ax=axes, ncols=ncols, nrows=nrows, projections=projections, **kwargs
        ) as axes:
            if nrows == ncols == 1:
                ax = axes
            else:
                ax = axes[0]
                bz_axes = axes[1]
            if case == "reg+zora":
                if target == "zora only":
                    logger.info(
                        "Plotting regular band structure without spin-orbit coupling."
                    )
                    bs = self.regular_bandstructure_zora
                    bs.plot(axes=ax, **kwargs)
                elif target == "two spin channels":
                    logger.info(
                        "Plotting regular band structure with both spin channels."
                    )
                    bs = self.regular_bandstructure_zora
                    self._plot_both_spin_channels(axes=ax, **kwargs)
            elif case == "reg+zora+soc":
                if target == "soc only":
                    logger.info(
                        "Plotting regular band structure with spin-orbit coupling."
                    )
                    bs = self.regular_bandstructure_soc
                    bs.plot(axes=axes, **kwargs)
                if target == "zora+soc":
                    logger.info("Plotting regular band structure with ZORA+SOC.")
                    bs = self.regular_bandstructure_soc
                    self._plot_zora_and_soc(axes=ax, **kwargs)
            if case == "mlk+zora":
                if target == "zora only":
                    logger.info(
                        "Plotting mulliken-projected band structure without spin-orbit coupling."
                    )
                    bs = self.mulliken_bandstructure_zora
                    bs.plot_all_species(axes=ax, **kwargs)
                elif target == "two spin channels":
                    logger.info(
                        "Plotting mulliken-projected band structure with both spin channels."
                    )
                    bs = self.mulliken_bandstructure_zora
                    self._plot_both_spin_channels(axes=ax, **kwargs)
            elif case == "mlk+zora+soc" and target == "soc only":
                logger.info(
                    "Plotting mulliken-projected band structure with spin-orbit coupling."
                )
                bs = self.mulliken_bandstructure_soc
                bs.plot_all_species(axes=ax, **kwargs)
            if show_BZ:
                self.plot_brillouin_zone(axes=bz_axes)

            self.reference = bs.spectrum.reference

        return axes

    def get_properties(self):
        """Utility function to print out band gap properties."""
        bs = (self._regular_bandstructure_soc or self._regular_bandstructure_zora) or (
            self._mulliken_bandstructure_soc or self._mulliken_bandstructure_zora
        )
        logger.info("Analyzing {} ...".format(bs))
        soc = bs.soc
        bg = bs.bandgap.soc if soc else bs.bandgap.scalar
        logger.info(
            "The band structure has been calculated {} spin-orbit-coupling.".format(
                "with" if soc else "without"
            )
        )
        logger.info("The band gap from the output file is {:.4f} eV large.".format(bg))
        if bs.control.spin == "none":
            bs.spectrum.print_bandgap_information()
        else:
            logger.info("Analyzing spin channel 1 ...")
            bs.spectrum.print_bandgap_information(spin="up")
            logger.info("Analyzing spin channel 2 ...")
            bs.spectrum.print_bandgap_information(spin="down")
