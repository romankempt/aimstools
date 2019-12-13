#!/usr/bin/env python
from distutils.core import setup

setup(
    name="AIMS_tools",
    version="0.0.5",
    author="Roman Kempt",
    author_email="roman.kempt@tu-dresden.de",
    description="A small toolbox to handle AIMS calculations.",
    long_description=open("README.md").read(),
    license="LGPLv3",
    url="https://github.com/romankempt/AIMS_tools",
    download_url="https://github.com/romankempt/AIMS_tools",
    packages=["AIMS_tools"],
    scripts=["bin/prepare_aims.py"],
    classifiers=[
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
