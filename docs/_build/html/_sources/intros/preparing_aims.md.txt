# Preparing AIMS calculations

The **preparation** module contains simple functionalities to set up the files needed for different tasks. This module is automatically called by the **aims_prepare.py** script.
Therefore, it sets up an ASE calculator object. In the current version of ASE (3.18), only single-points and forces are implemented for AIMS.
To circumvene this limitation, the input files are generated and modified.
The input is any file supported by the ASE containing (periodic) coordinate informations, e.g., xyz, cif or POSCAR.

If installed correctly, the aims_prepare.py script is added to your environment/bin folder and can be directly called from the command line:

Simply run:
```bash
aims_prepare.py geometry_inputfile [options]
```

The script has many different options. Show these with:
```bash
aims_prepare.py --help
```

This will print out:
```bash
usage: aims_prepare.py [-h] [-cluster CLUSTER] [-cost COST] [-m MEMORY]
                       [-n NODES] [-ppn PPN] [-wt WALLTIME] [-xc XC]
                       [-spin SPIN] [-tier TIER] [-basis BASIS]
                       [-k_grid K_GRID [K_GRID ...]] [-SOC SOC]
                       [-task TASK [TASK ...]] [-pbc PBC] [-vdw VDW]
                       geometry

positional arguments:
  geometry              Path to geometry

optional arguments:
  -h, --help            show this help message and exit
  -cluster CLUSTER      t3000 or taurus
  -cost COST            low, medium or high (automatically adjusts memory and
                        nodes)
  -m MEMORY, --memory MEMORY
                        memory in GB (default: 63)
  -n NODES, --nodes NODES
                        number of nodes (default: 4)
  -ppn PPN              processors per node (default: 20)
  -wt WALLTIME, --walltime WALLTIME
                        walltime in hours (default: 72)
  -xc XC                exchange correlation functional (default pbe)
  -spin SPIN            spin keyword (default none)
  -tier TIER            basis set tier (default 2)
  -basis BASIS          basis set type (default tight)
  -k_grid K_GRID [K_GRID ...]
                        k-points per reciprocal lattice direction for x, y, z
                        (default: 6 6 6)
  -SOC SOC              include spin-orbit coupling (default False)
  -task TASK [TASK ...]
                        list of task(s) to perform: None (default) = single
                        point; BS = band structure; DOS = (atom-projected)
                        density of states; GO = geometry optimisation
  -pbc PBC              Specify 2D periodic boundary conditions if your
                        coordinate file does not support 2D pbc with --pbc 2D.
  -vdw VDW              Add vdW correction, e.g., TS for
                        vdw_correction_hirshfeld or MBD for
                        many_body_dispersion (default: None).
```

The script will automatically generate the geometry.in, control.in and a submission file for the Taurus or T3000 cluster with a couple of senseful default options, but please refer to the AIMS manual for detailed options.
Don't forget that every AIMS calculation needs to be in its own separate directory. If you have a directory with a lot of structures, this can easily be done with:

```bash
for d in *.xyz; do (mkdir ${d%.xyz} && mv $d ${d%.xyz}/); done
```

An example would look like this:
```bash
aims_prepare.py MoS2.cif -cluster taurus -cost low -xc hse06 -basis tight -tier 2 -pbc 2D -vdw MBD -SOC True -task BS DOS
```

This will set up a band structure calculation following AFLOW conventions for the t3000 cluster with spin-orbit coupling and atom-projected densities of states on a 6x6x1 k-grid.
It employs the hse06 functional and adds the necessary options for functionals including exact HF exchange in AIMS.

Because .cif files are always three-dimensional, the -pbc 2D option is used to remove all z-components from the lattice basis. Note that AIMS does not support 2D boundary conditions, but simply uses a large lattice vector in the z-direction (for example 100 Angström).

The job can then simply be submitted via qsub on the *.sh script (t3000) or sbatch on the *.sh script (taurus). See taurus node limitations [here](https://doc.zih.tu-dresden.de/hpc-wiki/bin/view/Compendium/SystemTaurus).