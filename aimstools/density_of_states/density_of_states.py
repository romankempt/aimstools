from aimstools.misc import *
from aimstools.density_of_states.base import DOSBaseClass
from aimstools.density_of_states.total_dos import TotalDOS
from aimstools.density_of_states.atom_proj_dos import AtomProjectedDOS
from aimstools.density_of_states.species_proj_dos import SpeciesProjectedDOS


class DensityOfStates:
    """ Represents a collection of densities of states.

    The density of states class is a wrapper for total DOS, atom-projected DOS and species-projected DOS.
    To access these classes individually, see :class:`~aimstools.density_of_states.total_dos.TotalDOS`,
    :class:`~aimstools.density_of_states.atom_proj_dos.AtomProjectedDOS` and :class:`~aimstools.density_of_states.species_proj_dos.SpeciesProjectedDOS`.
    They are stored as properties (if calculated):

    >>> from aimstools import DensityOfStates as DOS
    >>> dos = DOS("/path/to/outputfile")
    >>> tdos_zora = dos.total_dos_zora
    >>> tdos_soc = dos.total_dos_soc
    >>> apdos_zora = dos.atom_dos_zora
    >>> apdos_soc = dos.atom_dos_soc
    >>> spdos_zora = dos.specices_dos_zora
    >>> spdos_soc = dos.species_dos_soc    

    The atom-projected DOS can produce all projections on the species and the total dos. It shares most of its methods with the species-projected DOS.

    An out-of-the box visualization can be done via:

    >>> dos.plot()

    Args:
        outputfile (str): Path to output file or output directory.

    """

    def __init__(self, outputfile) -> None:
        self.outputfile = outputfile
        self._tdos_zora = None
        self._tdos_soc = None
        self._apdos_zora = None
        self._apdos_soc = None
        self._spdos_zora = None
        self._spdos_soc = None
        self._set_classes()

    def _set_classes(self):
        self.base = DOSBaseClass(outputfile=self.outputfile)
        self.soc = self.base.control["include_spin_orbit"]
        self.methods = []
        if any(x in ["total dos", "total dos tetrahedron"] for x in self.base.tasks):
            self.methods.append("total")
            self._tdos_zora = TotalDOS(outputfile=self.outputfile, soc=False)
            if self.soc:
                self._tdos_soc = TotalDOS(outputfile=self.outputfile, soc=True)
        if any(
            x in ["atom-projected dos", "atom-projected dos tetrahedron"]
            for x in self.base.tasks
        ):
            self.methods.append("atom")
            self._apdos_zora = AtomProjectedDOS(outputfile=self.outputfile, soc=False)
            if self.soc:
                self._apdos_soc = AtomProjectedDOS(outputfile=self.outputfile, soc=True)
        if any(
            x in ["species-projected dos", "species-projected dos tetrahedron"]
            for x in self.base.tasks
        ):
            self.methods.append("species")
            self._spdos_zora = SpeciesProjectedDOS(
                outputfile=self.outputfile, soc=False
            )
            if self.soc:
                self._spdos_soc = SpeciesProjectedDOS(
                    outputfile=self.outputfile, soc=True
                )

    def __repr__(self):
        return "{}(outputfile={}, spin_orbit_coupling={})".format(
            self.__class__.__name__, repr(self.outputfile), self.soc
        )

    @property
    def total_dos_zora(self):
        """ Returns :class:`~aimstools.density_of_states.total_dos.TotalDOS` without spin-orbit coupling."""
        return self._tdos_zora

    @property
    def total_dos_soc(self):
        """ Returns :class:`~aimstools.density_of_states.total_dos.TotalDOS` with spin-orbit coupling."""
        return self._tdos_soc

    @property
    def atom_dos_zora(self):
        """ Returns :class:`~aimstools.density_of_states.atom_proj_dos.AtomProjectedDOS` without spin-orbit coupling."""
        return self._apdos_zora

    @property
    def atom_dos_soc(self):
        """ Returns :class:`~aimstools.density_of_states.atom_proj_dos.AtomProjectedDOS` with spin-orbit coupling."""
        return self._apdos_soc

    @property
    def species_dos_zora(self):
        """ Returns :class:`~aimstools.density_of_states.species_proj_dos.SpeciesProjectedDOS` without spin-orbit coupling."""
        return self._spdos_zora

    @property
    def species_dos_soc(self):
        """ Returns :class:`~aimstools.density_of_states.species_proj_dos.SpeciesProjectedDOS` with spin-orbit coupling."""
        return self._spdos_soc

    def plot(self, **kwargs):
        base = self.base
        kwargs["flip"] = kwargs.get("flip", False)
        spin = False if base.control["spin"] == "none" else True
        fig, axes = plt.subplots(
            (len(self.methods)), 1, figsize=(5, 3 * len(self.methods))
        )
        i = 0
        if "total" in self.methods:
            plt.sca(axes[i])
            if self.soc:
                axes[i] = self.total_dos_soc.plot(
                    color=mutedblack, axes=axes[i], **kwargs
                )
            else:
                axes[i] = self.total_dos_zora.plot(
                    color=mutedblack, axes=axes[i], spin="dn", **kwargs
                )
                if spin:
                    axes[i] = self.total_dos_zora.plot(
                        color=mutedblack, axes=axes[i], spin="up", **kwargs
                    )
            axes[i].set_title("Total DOS")
            i += 1
        if "species" in self.methods:
            plt.sca(axes[i])
            if self.soc:
                axes[i] = self.species_dos_soc.plot_all_species(axes=axes[i], **kwargs)
            else:
                axes[i] = self.species_dos_zora.plot_all_species(
                    axes=axes[i], spin="dn", **kwargs
                )
                if spin:
                    axes[i] = self.species_dos_zora.plot_all_species(
                        axes=axes[i], spin="up", **kwargs
                    )
            axes[i].set_title("Species-Projected DOS")
            i += 1
        if "atom" in self.methods:
            plt.sca(axes[i])
            if self.soc:
                axes[i] = self.atom_dos_soc.plot_all_angular_momenta(
                    axes=axes[i], **kwargs
                )
            else:
                axes[i] = self.atom_dos_zora.plot_all_angular_momenta(
                    axes=axes[i], spin="dn", **kwargs
                )
                if spin:
                    axes[i] = self.atom_dos_zora.plot_all_angular_momenta(
                        axes=axes[i], spin="up", **kwargs
                    )
            axes[i].set_title("Atom-Projected DOS")
        plt.tight_layout(pad=2)
        plt.show()
