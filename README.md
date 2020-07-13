# AIMS_tools

This library contains a personal collection of scripts to handle AIMS calculations. It's mainly meant for private use or to be shared with students and colleagues.

## Installation

I recommend using Anaconda to manage package dependencies and updates. See [here](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

Directly install from git:

```bash
pip install git+https://github.com/romankempt/AIMS_tools
```

Or download from git and install from the zip.

If you are using AIMS_tools locally, you need to specify the path to the aims species (basis sets) and executable in your environment variables (e.g., in the .bashrc). On Taurus, loading the module will automatically take care of that.

```bash
export AIMS_SPECIES_DIR="path/to/directory"
export AIMS_EXECUTABLE="aims.mpi.x"
```

- On t3000, this one is "/chemsoft/FHI-aims/stable/species_defaults/"
- On taurus, this one is "/projects/m_chemie/FHI-aims/aims_200112/species_defaults/"


## Documentation
The documentation is now available at [ReadTheDocs](https://aims-tools.readthedocs.io/en/master/).


## Requirements

- [Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/) ase 3.19 or higher
- [Space Group Libary](https://atztogo.github.io/spglib/python-spglib.html) spglib
- [Scientific Python](https://www.scipy.org/) scipy (including numpy, matplotlib and pandas)
- [seaborn](https://seaborn.pydata.org/)
- [networkx](https://networkx.github.io/documentation/stable/install.html)
