""" Definition of density of states analysis and plotting functions. """
from aimstools.misc import *

from aimstools.density_of_states.density_of_states import DensityOfStates
from aimstools.density_of_states.total_dos import TotalDOS
from aimstools.density_of_states.atom_proj_dos import AtomProjectedDOS
from aimstools.density_of_states.species_proj_dos import SpeciesProjectedDOS


__all__ = ["DensityOfStates", "TotalDOS", "AtomProjectedDOS", "SpeciesProjectedDOS"]

allow_plotting = os.environ.get("ALLOW_PLOTTING")
if allow_plotting == False:
    raise Exception(
        "The plotting functionalities are not available on this system, e.g., if you are running aimstools on a cluster."
    )
