"""AIMS_tools"""

import sys
from distutils.version import LooseVersion

if sys.version_info[0] == 2:
    raise ImportError("Requires Python3. This is Python2.")

__version__ = "0.1.4"
