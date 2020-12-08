# How to set up an FHI-aims environment


This guide walks you through the steps to set up an FHI-aims python environment via Anaconda similar to the ones that I have set up on our HPC systems.
The guide is meant for newcomers and people unfamiliar with python. You do not need to do set up all external libraries (e.g., `BoltzTraP2`) right away
to use `aimstools`. You can also do that later.

The guide is targetting Unix-based systems. If you work with Windows, please set up the [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/install-win10). I would recommend version 2 of the WSL for speed and possibly graphical applications.

## Install a python package management system

You can install a python package management system of your choosing, but I strongly recommend Anaconda. The guide will also be centered around Anaconda.

A Miniconda installation is sufficient. You can download the `bash executable .sh` from [here](https://docs.conda.io/en/latest/miniconda.html). WSL users, please choose the `Linux` downloadables. Please choose the newest `Python 3` version, for example `Python 3.8`. Once finished downloading, please install in your `$HOME` by executing the `Miniconda*.sh` script, possibly with Admin rights.

When the installation is finished, it will ask you to initialize conda in your current shell. Please accept this. If you forgot to accept it, you might have to locate the conda executable and run `conda init bash`. This will make some minor changes to your `~\.bashrc`. Restart your shell (`source ~\.bashrc`) or open a new one to take effect. You should see the currently active conda environment in brackets in the beginning of your line like this: `(base) yourname@yourmachine:/some/path`.

## Create a virtual environment

It's good practice to keep all python libraries relevant to a specific project in a specific virtual environment, e.g., such that you have one environment to work with `Crystal17`, another one to work with `AMS 2020` and one to work with `FHI-aims`. This avoids dependency conflicts and easily allows modifications without breaking other installations. This also allows to have python 2 and python 3 installations in separate environments, as well as to have separate development environments.

To create an FHI-aims environment, type:
```bash
conda create --name aims-env python=3.8
```

This sets up a conda environment with the name `aims-env` and python version 3.8.
Activate this environment by:
```bash
conda activate aims-env
```

This changes from the `(base)` environment to the `(aims-env)` environment. If aims-env is your favorite one, you can put the line `conda activate aims-env` in your `~\.bashrc` such that it is activated by default.

Then, install a couple of useful packages:
```bash
conda install -c conda-forge git pip zip cmake 
conda install -c conda-forge numpy scipy matplotlib ase
```

The basic stuff is installed and you are good to go. You can now open a python shell by typing `python`. When you are in a python shell, this is indicated by `>>>` in front of the line. See that everything works:

```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> import ase
``` 

## Installing aimstools

When everything is set up, installing (and updating) aimstools is as easy as:

```
pip install git+https://github.com/romankempt/aimstools
```

Be sure to do that in your environment `aims-env`. You should now have access to the command line tools:
```bash
which aims_prepare
whereis aims_plot
```

If the path to these is not showing in your shell, something went wrong during the installation. Try to import the library:
```python
>>> import aimstools
>>> aimstools.__version__
```

To use all modules of aimstools locally, the minimum you need are the FHI-aims basis sets (species defaults). Please download these from the FHI-aims gitlab server or one of our HPC systems. Add `export AIMS_SPECIES_DIR="path/to/species_defaults"` to your `~\.bashrc`.

## Installing FHI-vibes

FHI-vibes is an excellent interface to run relaxations, phonon calculations and molecular dynamics. It is the recommended way by aimstools (so by me) to perform these tasks with FHI-aims.

Please follow the instructions [here](https://vibes-developers.gitlab.io/vibes/Installation/) to install `FHI-vibes`.

The only additional dependency you will need is a `fortran-compiler`, which you can install via `conda install -c conda-forge fortran-compiler`.
Then run:
```bash
pip install fhi-vibes
```

## Optional configurations

The other environment variables are optional. To run FHI-aims calculations locally or use FHI-aims through the ASE, you also need an FHI-aims binary. Then, specify in your `~\.bashrc` the environment variables `export AIMS_EXECUTABLE=/path/to/executable`. If you call the FHI-aims binary from a python shell, it does not necessarily inherit all environment variables. Thus, it is a good idea to execute a bash script through python that contains environment information. Name this bash script `run_aims.sh` and specify:

```bash
#!/bin/bash -l

ulimit -s unlimited
export OMP_NUM_THREADS=1

mpirun /path/to/FHIaims/binary
```

Add the location of `run_aims.sh` to your `$PATH`. Also, specify the ASE environment variable `export ASE_AIMS_COMMAND=/path/to/run-aims.sh`.

You can also specify paths to custom submission scripts via the environment variables `$AIMS_SLURM_TEMPLATE` and `$VIBES_SLURM_TEMPLATE`. These should be simple text files looking like this:

`AIMS_SLURM_TEMPLATE`:
```bash
#!/bin/bash
#SBATCH -J {jobname}                 # job name

/path/to/run-aims.sh > aims.out
```

`VIBES_SLURM_TEMPLATE`
```bash
#!/bin/bash
#SBATCH -J {jobname}                 # job name

vibes run {task} > log.{task}
```

The fields `{jobname}` and `{task}` will be filled in by aimstools.