import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="AIMS_tools-Roman-Kempt",  # Replace with your own username
    version="0.0.1",
    author="Roman Kempt",
    author_email="roman.kempt@tu-dresden.de",
    description="A small toolbox to handle AIMS calculations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="LGPLv3",
    url="https://github.com/romankempt/AIMS_tools",
    download_url="https://github.com/romankempt/AIMS_tools",
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=["ase", "spglib", "numpy", "matplotlib", "seaborn"],
    dependency_links=[
        "https://github.com/atztogo/spglib",
        "https://gitlab.com/ase/ase.git",
        "https://github.com/mwaskom/seaborn",
    ],
    python_requires=">=3.6",
)
