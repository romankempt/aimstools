# aimstools

This library contains a personal collection of scripts to handle FHI-aims calculations. It's mainly meant for private use or to be shared with students and colleagues.

## Installation

I recommend using Anaconda to manage package dependencies and updates. See [here](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

Directly install from git:

```bash
pip install git+https://github.com/romankempt/aimstools
```

Or download from git and install from the zip.

If you are using AIMS_tools locally, you need to specify the path to the aims species (basis sets) and executable in your environment variables (e.g., in the .bashrc).
On our HPC systems, my module environments take care of that.

```bash
export AIMS_SPECIES_DIR="path/to/directory"
export AIMS_EXECUTABLE="aims.mpi.x"
```


## Documentation
The documentation is now available at [ReadTheDocs](https://aims-tools.readthedocs.io/en/master/).


## Requirements

- [Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/) ase 3.19 or higher
- [Space Group Libary](https://atztogo.github.io/spglib/python-spglib.html) spglib
- [Scientific Python](https://www.scipy.org/) scipy (including numpy, matplotlib and pandas)
- [seaborn](https://seaborn.pydata.org/)
- [networkx](https://networkx.github.io/documentation/stable/install.html)
