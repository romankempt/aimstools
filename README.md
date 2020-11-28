# Tools for FHI-aims

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/romankempt/aimstools/HEAD)

This library contains a personal collection of scripts to handle FHI-aims calculations. It's mainly meant for private use or to be shared with students and colleagues.

## Installation

I recommend using Anaconda to manage package dependencies and updates. See [here](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

You can directly install from git:

```bash
conda install -c conda-forge git pip
pip install git+https://github.com/romankempt/aimstools
```

If you are using aimstools locally, you need to specify the path to the FHI-aims species (basis sets) in your environment variables (e.g., in the .bashrc) in order to use the file preparation utilities. Furthermore, you may need to specify the path to the FHI-aims executable and paths to slurm submission script templates if requested.

On our HPC systems, my module environments take automatically take care of these things.

```bash
export AIMS_SPECIES_DIR="path/to/directory"
export AIMS_EXECUTABLE="aims.mpi.x"
```

## Documentation
The documentation is now available at [ReadTheDocs](https://aims-tools.readthedocs.io/en/master/).

## Requirements

- [Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/)
- [Space Group Libary](https://atztogo.github.io/spglib/python-spglib.html)
- [Scientific Python](https://www.scipy.org/) scipy (including numpy, matplotlib and pandas)
- [networkx](https://networkx.github.io/documentation/stable/install.html)

## Further recommended libraries

- [FHI-vibes](https://vibes-developers.gitlab.io/vibes/) - An excellent interface for relaxations, phonon calculations and molecular dynamics.
- [BoltzTraP2](https://gitlab.com/sousaw/BoltzTraP2) - Boltzmann transport theory to calculate electronic transport properties.
- [2D-Interface-Builder](https://github.com/AK-Heine/2D-Interface-Builder) - My tool to build heterostructure interfaces via coincidence lattice theory.

## Testing

Install [pytest](https://docs.pytest.org/en/stable/) to run a simple set of tests to see that everything installed correctly. Clone the repository and navigate to the directory, then run:

```bash
pytest -v tests/
```