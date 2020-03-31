# Command line tools

## Preparing AIMS calculations

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



## Plotting band structures and densities of states

The **bandstructure** and **dos** module are wrapped via the **multiplots** module in the **aims_plot** command line tool. The syntax follows the *multiplots.combine()* function.

```bash
usage: aims_plot [-h] [-nrows NROWS] [-ncols NCOLS] [-ratios RATIOS [RATIOS ...]] [-titles TITLES [TITLES ...]] [-s KEY=VALUE [KEY=VALUE ...]] [-w WRITE]
                 directories [directories ...]

positional arguments:
  directories           List of directories to plot

optional arguments:
  -h, --help            show this help message and exit
  -nrows NROWS          Number of rows
  -ncols NCOLS          Number of rows
  -ratios RATIOS [RATIOS ...]
                        List of ratios
  -titles TITLES [TITLES ...]
                        List of titles
  -s KEY=VALUE [KEY=VALUE ...], --set KEY=VALUE [KEY=VALUE ...]
                        Set arbitrary number of key-value pairs for plotting. Pay attention that there is no space before and after the '='. Example: --set color='red'
  -w WRITE, --write WRITE
                        Saves figure with given filename.
```

For example:

```bash
aims_plot bandstructure1/ dos1/ bandstructure2/ dos2/ -nrows 1 -ncols 4 -ratios 3 1 3 1 -s linewidth=2 -w test.png
```