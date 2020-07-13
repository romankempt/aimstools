#!/usr/bin/env python
from distutils.core import setup

import re

VERSIONFILE = "AIMS_tools/__init__.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    version = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(
    name="AIMS_tools",
    version=version,
    author="Roman Kempt",
    author_email="roman.kempt@tu-dresden.de",
    description="A small toolbox to handle AIMS calculations.",
    long_description=open("README.md").read(),
    license="LGPLv3",
    url="https://github.com/romankempt/AIMS_tools",
    download_url="https://github.com/romankempt/AIMS_tools",
    packages=["AIMS_tools"],
    scripts=[
        "bin/aims_prepare",
        "bin/aims_sort",
        "bin/aims_plot",
        "bin/aims_kconvergence",
        "bin/aims_standardize",
    ],
    install_requires=["spglib", "numpy", "scipy", "matplotlib", "ase", "networkx"],
    classifiers=[
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
