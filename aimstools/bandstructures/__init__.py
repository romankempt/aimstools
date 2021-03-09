""" Definition of bandstructure analysis and plotting functions. """

from aimstools.misc import *
from aimstools.bandstructures.bandstructure import BandStructure
from aimstools.bandstructures.regular_bandstructure import RegularBandStructure
from aimstools.bandstructures.brillouinezone import BrillouineZone
from aimstools.bandstructures.mulliken_bandstructure import MullikenBandStructure

import os

__all__ = [
    "BandStructure",
    "BrillouineZone",
    "MullikenBandStructure",
    "RegularBandStructure",
]


allow_plotting = os.environ.get("AIMSOOLS_ALLOW_PLOTTING")
if allow_plotting == "FALSE":
    raise Exception(
        "The plotting functionalities are not available on this system, e.g., if you are running aimstools on a cluster."
    )

