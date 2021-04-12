#!/usr/bin/env python
from distutils.core import setup
from setuptools import find_packages

import re

VERSIONFILE = "aimstools/__init__.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    version = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


setup(
    name="aimstools",
    version=version,
    author="Roman Kempt",
    author_email="roman.kempt@tu-dresden.de",
    description="Tools for FHI-aims",
    long_description=open("README.md").read(),
    license="LGPLv3",
    url="https://github.com/romankempt/aimstools",
    download_url="https://github.com/romankempt/aimstools",
    packages=find_packages(
        exclude=[
            "*.tests",
            "*.tests.*",
            "tests.*",
            "tests",
            "WIP",
            "pictures",
            "examples",
            "docs",
            "*__pycache__*",
            "*vscode*",
        ]
    ),
    package_data={"": ["*.mplstyle"]},
    scripts=["bin/aims_prepare", "bin/aims_plot", "bin/aims_workflow"],
    install_requires=[
        "spglib",
        "numpy",
        "scipy",
        "matplotlib",
        "ase",
        "networkx",
        "pandas",
        "pretty_errors",
        "rich",
        "pyyaml",
        "typer",
        "phonopy",
    ],
    classifiers=[
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
