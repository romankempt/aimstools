# AIMS_tools
This library contains a personal collection of scripts to handle AIMS calculations. It's mainly meant for private use or to be shared with students and colleagues.

## Currently implemented features
- **Running AIMS:** prepare_aims.py can set up the files for different tasks employing the ASE (single point, geometry optimisation, band structure calculation, atom-projected density of states)
- **Plotting functionalities:**
The plotting modules are designed for combinatorial plotting with a lot of flexibility. It's very easy to overlay, compare and arrange plots in different fashions with the in-built classes.
    > **bandstructure.py** can handle band structure plots with scalar relativity, SOC, and mulliken-projection ("fatbands"). **To do:** improve fatband projections, include plotting with spin

    > **dos.py** handles atom-projected densities of states. Summing up different DOS contributions or to get the total DOS is easily done.

    > **multiplots.py** contains exemplary functions to combine plots in different fashions, for example overlaying ZORA and ZORA+SOC bandstructures or bandstructures and DOS.

    > **To do:** instead of making every module in-line executable, I will provide a superscript that automatically detects the calculation type and provides an easy plotting functionality.

## Installation
I recommend using Anaconda to manage package dependencies and updates. See [here](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).
For simplicity, download my [ase-env](https://anaconda.org/romankempt/ase-env/files) from the anaconda cloud. This should contain all necessary libraries and a couple more.

Simply download the zip from github and unpack it.
Then run:

```bash
python setup.py install
```

Set the path to the aims species (basis sets) in your environment variables (e.g., in the .bashrc):

```bash
export AIMS_SPECIES_DIR="path/to/directory"
```

- On t3000, this one is "/chemsoft/FHI-aims/stable/species_defaults/"
- On taurus, this one is "/projects/m_chemie/FHI-aims/FHI-aims_4_Roman/aimsfiles-master/species_defaults/"


## Documentation
I'm working on the documentation. You can find it under docs/_build/html/index.html.


## Requirements
- [Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/) ase 3.18.1 or higher
- [Space Group Libary](https://atztogo.github.io/spglib/python-spglib.html) spglib
- [scipy](https://www.scipy.org/)
- [seaborn](https://seaborn.pydata.org/)
- numpy
- pandas

For simplicity, download my [ase-env](https://anaconda.org/romankempt/ase-env/files) from the anaconda cloud. This should contain all necessary libraries and a couple more.