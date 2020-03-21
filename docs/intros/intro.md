# AIMS_tools

This library contains a personal collection of scripts to handle AIMS calculations. It's mainly meant for private use or to be shared with students and colleagues.


## Installation

I recommend using Anaconda to manage package dependencies and updates. See [here](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

Simply download the zip from github. Then run:

```bash
pip install AIMS_tools-master.zip
```

Set the path to the aims species (basis sets) in your environment variables (e.g., in the .bashrc):

```bash
export AIMS_SPECIES_DIR="path/to/directory"
```

On taurus, this one is "/projects/m_chemie/FHI-aims/aims_200112/species_defaults/".


## Testing

To run tests, install the pytest library. Navigate to the AIMS_tools directory and run:

```bash
pytest -v tests
```


## Documentation
The documentation is now available at [ReadTheDocs](https://readthedocs.org/projects/aims-tools/).


## Requirements

- [Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/) ase 3.18.1 or higher
- [Space Group Libary](https://atztogo.github.io/spglib/python-spglib.html) spglib
- [Scientific Python](https://www.scipy.org/) scipy (including numpy, matplotlib and pandas)
- [seaborn](https://seaborn.pydata.org/)
- [networkx](https://networkx.github.io/documentation/stable/install.html)
