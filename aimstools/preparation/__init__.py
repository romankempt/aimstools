""" Utilities to prepare aims calculations for different tasks. """

from aimstools.misc import *
from aimstools.preparation.aims_setup import FHIAimsSetup
from aimstools.preparation.vibes_setup import FHIVibesSetup

import os


assert (
    os.getenv("AIMS_SPECIES_DIR") != None
), "FHI-aims species defaults not found iin environment (set $AIMS_SPECIES_DIR)."

if os.getenv("AIMS_SLURM_TEMPLATE") == None:
    logger.info(
        "FHI-aims slurm template not found in environment (set $AIMS_SLURM_TEMPLATE)."
    )

if os.getenv("VIBES_SLURM_TEMPLATE") == None:
    logger.info(
        "FHI-aims slurm template not found in environment (set $AIMS_SLURM_TEMPLATE)."
    )

if os.getenv("AIMS_EXECUTABLE") == None:
    logger.info("FHI-aims executable not found in environment (set $AIMS_EXECUTABLE).")

if os.getenv("ASE_AIMS_COMMAND") == None:
    logger.info(
        "ASE command to run FHI-aims not found in environment (set $ASE_AIMS_COMMAND)."
    )


__all__ = ["FHIAimsSetup", "FHIVibesSetup"]

