# Preparing AIMS calculations

The **preparation** module contains simple functionalities to set up the files needed for different tasks. This module is wrapped in the **aims_prepare** command line tool.
The input is any file supported by the ASE containing (periodic) coordinate informations, e.g., xyz, cif or POSCAR.

```bash
aims_prepare geometry_inputfile [options]
```

The script has different options. Show these with:
```bash
aims_prepare.py --help

usage: aims_prepare [-h] [-cluster CLUSTER] [-cost COST] [-xc XC] [-spin SPIN] [-tier TIER] [-basis BASIS] [-k_grid K_GRID [K_GRID ...]] [-task TASK [TASK ...]] geometry

positional arguments:
  geometry              Path to geometry

optional arguments:
  -h, --help            show this help message and exit
  -cluster CLUSTER      t3000 or taurus (default taurus)
  -cost COST            low, medium or high (automatically adjusts memory and nodes)
  -xc XC                exchange correlation functional (default pbe)
  -spin SPIN            spin keyword (default none)
  -tier TIER            basis set tier (default 2)
  -basis BASIS          basis set type (default tight)
  -k_grid K_GRID [K_GRID ...]
                        k-points per reciprocal lattice direction for x, y, z (default: 6 6 6)
  -task TASK [TASK ...]
                        list of task(s) to perform: None (default) = single point; BS = band structure; fatBS = mulliken-projected band structure; DOS = (atom-projected) density of states; GO = geometry optimisation; phonons = phonopy
                        interface
```

The script will automatically generate the geometry.in, control.in and a submission file for the Taurus or T3000 cluster with a couple of senseful default options, but please refer to the AIMS manual for detailed options.

An example would look like this:
```bash
aims_prepare MoS2.cif -cluster taurus -cost low -xc hse06 -basis tight -tier 2 -task BS DOS
```

The job can then simply be submitted via qsub on the *.sh script (t3000) or sbatch on the *.sh script (taurus). See taurus node limitations [here](https://doc.zih.tu-dresden.de/hpc-wiki/bin/view/Compendium/SystemTaurus).