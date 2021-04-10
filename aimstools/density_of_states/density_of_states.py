from aimstools.misc import *
from aimstools.density_of_states.base import DOSBaseClass
from aimstools.density_of_states.total_dos import TotalDOS
from aimstools.density_of_states.atom_proj_dos import AtomProjectedDOS
from aimstools.density_of_states.species_proj_dos import SpeciesProjectedDOS


class DensityOfStates(AtomProjectedDOS, SpeciesProjectedDOS, TotalDOS, DOSBaseClass):
    """ High-level wrapper for densities of states.

    The density of states class is a wrapper for total DOS, atom-projected DOS and species-projected DOS.
    To access these classes individually, see :class:`~aimstools.density_of_states.total_dos.TotalDOS`,
    :class:`~aimstools.density_of_states.atom_proj_dos.AtomProjectedDOS` and :class:`~aimstools.density_of_states.species_proj_dos.SpeciesProjectedDOS`.
    
    The method resolution order is AtomProjectedDOS > SpeciesProjectedDOS > TotalDOS. All methods of these classes are inherited.

    >>> from aimstools import DensityOfStates as DOS
    >>> dos = DOS("/path/to/outputfile")
    >>> dos.plot()

    Args:
        outputfile (str): Path to output file or output directory.
    """

    def __init__(self, outputfile) -> None:
        DOSBaseClass.__init__(self, outputfile=outputfile)
        soc = self.control.include_spin_orbit
        if any(
            x in ["atom-projected dos", "atom-projected dos tetrahedron"]
            for x in self.tasks
        ):
            logger.info("Found results for atom-projected density of states.")
            AtomProjectedDOS.__init__(self, outputfile=self.outputfile, soc=soc)
        elif any(
            x in ["species-projected dos", "species-projected dos tetrahedron"]
            for x in self.tasks
        ):
            logger.info("Found results for species-projected density of states.")
            SpeciesProjectedDOS.__init__(self, outputfile=self.outputfile, soc=soc)
        elif any(x in ["total dos", "total dos tetrahedron"] for x in self.tasks):
            logger.info("Found results for total density of states.")
            TotalDOS.__init__(self, outputfile=self.outputfile, soc=soc)

    def plot(self, axes=None, **kwargs):
        if any(
            x in ["atom-projected dos", "atom-projected dos tetrahedron"]
            for x in self.tasks
        ):
            self.plot_all_species(axes=axes, **kwargs)
        elif any(
            x in ["species-projected dos", "species-projected dos tetrahedron"]
            for x in self.tasks
        ):
            self.plot_all_species(axes=axes, **kwargs)
        elif any(x in ["total dos", "total dos tetrahedron"] for x in self.tasks):
            self.plot(axes=axes, **kwargs)

        return axes
