from aimstools.misc import *
from aimstools.density_of_states.base import DOSBaseClass
from aimstools.density_of_states.total_dos import TotalDOS
from aimstools.density_of_states.atom_proj_dos import AtomProjectedDOS
from aimstools.density_of_states.species_proj_dos import SpeciesProjectedDOS


class DensityOfStates(DOSBaseClass):
    """Represents a collection of density of states calculations.

    The density of states class is a wrapper for total DOS, atom-projected DOS and species-projected DOS.
    To access these classes individually, see :class:`~aimstools.density_of_states.total_dos.TotalDOS`,
    :class:`~aimstools.density_of_states.atom_proj_dos.AtomProjectedDOS` and :class:`~aimstools.density_of_states.species_proj_dos.SpeciesProjectedDOS`.
    
    >>> from aimstools import DensityOfStates as DOS
    >>> dos = DOS("/path/to/outputfile")
    >>> dos.plot()

    Args:
        outputfile (str): Path to output file or output directory.
    """

    def __init__(self, outputfile) -> None:
        DOSBaseClass.__init__(self, outputfile=outputfile)
        soc = self.control.include_spin_orbit
        self._total_dos = None
        self._species_dos = None
        self._atom_dos = None
        if any(
            x in ["atom-projected dos", "atom-projected dos tetrahedron"]
            for x in self.tasks
        ):
            logger.info("Found results for atom-projected density of states.")
            self._atom_dos = AtomProjectedDOS(outputfile=self.outputfile, soc=soc)
        elif any(
            x in ["species-projected dos", "species-projected dos tetrahedron"]
            for x in self.tasks
        ):
            logger.info("Found results for species-projected density of states.")
            self._species_dos = SpeciesProjectedDOS(outputfile=self.outputfile, soc=soc)
        elif any(x in ["total dos", "total dos tetrahedron"] for x in self.tasks):
            logger.info("Found results for total density of states.")
            self._total_dos = TotalDOS(outputfile=self.outputfile, soc=soc)

    @property
    def atom_projected_dos(self):
        "Returns :class:`~aimstools.density_of_states.atom_proj_dos.AtomProjectedDOS` if calculated."
        if self._atom_dos == None:
            raise Exception("Atom-projected density of states was not calculated.")
        return self._atom_dos

    @property
    def species_projected_dos(self):
        "Returns :class:`~aimstools.density_of_states.species_proj_dos.SpeciesProjectedDOS` if calculated."
        if self._species_dos == None:
            raise Exception("Species-projected density of states was not calculated.")
        return self._species_dos

    @property
    def total_dos(self):
        "Returns :class:`~aimstools.density_of_states.total_dos.TotalDOS` if calculated."
        if self._total_dos == None:
            raise Exception("Total density of states was not calculated.")
        return self._total_dos

    @property
    def dos(self):
        "Returns density of states in order AtomProjectedDOS > SpeciesProjectedDOS > TotalDOS."
        return self._atom_dos or self.species_projected_dos or self.total_dos

    def plot(self, axes=None, **kwargs):
        if any(
            x in ["atom-projected dos", "atom-projected dos tetrahedron"]
            for x in self.tasks
        ):
            self.dos.plot_all_species(axes=axes, **kwargs)
        elif any(
            x in ["species-projected dos", "species-projected dos tetrahedron"]
            for x in self.tasks
        ):
            self.dos.plot_all_species(axes=axes, **kwargs)
        elif any(x in ["total dos", "total dos tetrahedron"] for x in self.tasks):
            self.dos.plot(axes=axes, **kwargs)

        return axes
