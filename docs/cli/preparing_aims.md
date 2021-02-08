# Preparing FHI-aims calculations

The **preparation** module contains functionalities to set up the files needed for different tasks. This module is wrapped in the **aims_prepare** command line tool.
The input is any file supported by the ASE containing (periodic) coordinate informations, e.g., .xyz, .cif or POSCAR.

```bash
aims_prepare geometry_inputfile [options]
```

The script has different options. Show these with:
```bash
aims_prepare.py --help
```

| option                        | function                                                                                       | default             |
| ----------------------------- | ---------------------------------------------------------------------------------------------- | ------------------- |
| `--xc`                        | Sets exchange-correlation functional and additional keywords, e.g., for hybrid functionals.    | PBE                 |
| `--spin`                      | Sets spin to none or collinear and adds additional keywords for the initial spin moment.       | none                |
| `-t, --tier`                  | Sets basis set tier 1, 2, 3 or 4.                                                              | 1                   |
| `-b, --basis`                 | Sets basis set integration grids. Can be light, tight or intermediate depending on species.    | tight               |
| `-k, -k_density`              | Chooses k-grid based on line k-point density. Preferred over -k_grid .                         | 5 points / Angström |
| `--k_grid`                    | Explicitly sets number of k-points per reciprocal lattice direction for x, y and z.            | None                |
| `-j, --jobs, --task, --tasks` | Sets up different types of FHI-aims or FHI-vibes tasks. See more below.                        | None                |
| `-v, --verbose, -vv, -vvv`    | Sets verbosity level depending on number of "v". Verbosity levels are warning, info and debug. | 0                   |
| `-s, --standardize`           | Standardize structure via spglib with enforced axes order.                                     | False               |
| `-f`                          | Force overwrite of existing files.                                                             | False               |

The supported tasks for the option `-j` are:

| keyword      | task                                                                                                                                  |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| `bs`         | Sets up regular bandstructure calculation with Setyawan-Curtarolo convention regarding Brillouine zone sampling.                      |
| `fatbs`      | Sets up mulliken bandstructure calculation with Setyawan-Curtarolo convention regarding Brillouine zone sampling.                     |
| `dos`        | Sets up density of states calculation. By default, atom-projected and total density of states with the tetrahedron method are chosen. |
| `go`         | Sets up relaxation with FHI-vibes.                                                                                                    |
| `phonons`    | Sets up phonon calculation with FHI-vibes.                                                                                            |
| `absorption` | Sets up calculation of the dielectric tensor.                                                                                         |

The options `bs`, `dos` and `fatbs` are additive, whereas the options `go` and `phonons` are exclusive.

An examplary call looks like this:
```bash
aims_prepare MoS2.cif -basis tight -tier 1 -task BS DOS
```

The submission scripts are generated from templates which can be modified by setting the environment variables `$AIMS_SLURM_TEMPLATE` and `$VIBES_SLURM_TEMPLATE`.

## Modifying the control.in file

The script produces an ASE-generated `control.in` with often-used keywords being included, but commented out. For their function, please consult the FHI-aims manual.

```vim
xc      pbe
# include_spin_orbit                        <-- uncomment to include spin-orbit coupling
#        vdw_correction_hirshfeld           <-- uncomment to use TS correction
#        many_body_dispersion               <-- uncomment to use MBD correction
#        many_body_dispersion_nl            <-- uncomment to use MBD-nl correction
relativistic    atomic_zora scalar
spin        None
k_grid      15 15 1

### SCF settings
adjust_scf       always          3          
sc_iter_limit    100
# frozen_core_scf        .true.             <-- uncomment to freeze core states for heavy elements
# charge_mix_param       0.02               <-- uncomment to fix charge_mix_param (otherwise set by adjust_scf)
# occupation_type   gaussian    0.1         <-- uncomment to fix smearing (otherwise set by adjust_scf)
# sc_accuracy_rho        1E-6               <-- uncomment to fix density convergence accuracy (otherwise set by adjust_scf)
# elsi_restart  read_and_write 1000         <-- uncomment to read and write SCF restart files (thse take up a lot of space!)
```

## Modifying the relaxation.in file

If you perform a relaxation as task, the script produces a template `relaxation.in` file for optimization with FHI-vibes. Please work through the [FHI-vibes Relaxation Tutorial](https://vibes-developers.gitlab.io/vibes/Tutorial/1_geometry_optimization/) to understand the function of all keywords. I've set defaults according to the settings that typically work for me.

```vim
[calculator]
name:                          aims

[calculator.parameters]
xc:                            pbe
many_body_dispersion_nl:                <-- MBD-nl is default and takes no further arguments. You can change to any other dispersion correction.
tier: 1                                 <-- Tier 1 is default and typically sufficient for dense, periodic systems.
sc_accuracy_rho: 1e-6                   <-- For converged geometries, the density accuracy should be one order of magnitude higher than the force accuracy.
sc_iter_limit: 100                      <-- A typical SCF should not take more than 100 steps, unless you are dealing with magnetic systems.

[calculator.kpoints]
density: 5.00                           <-- Semiconductors and insulators do not need a high k-point density, but metals and semimetals do.

[calculator.basissets]
default:                       light    <-- For initial relaxations, choose light. To obtain converged geometries, you will need tight.

[calculator.socketio]
port: auto                              

[relaxation]
driver:                        BFGS
fmax:                          1e-3     <-- For initial relaxations, choose a higher value like 1e-2 or 5e-3. To obtain converged geometries, choose 1e-3 or lower.
unit_cell:                     True     <-- Enable relaxation of the unit cell.
fix_symmetry:                  False    <-- Fix space group symmetry.
hydrostatic_strain:            False    <-- Only the volume of the unit cell can change, but not the shape.
constant_volume:               False    <-- Only the shape of the unit cell can change, but not the volume.
scalar_pressure:               0.0      <-- Add a constant pressure for the relaxation.
decimals:                      12       
symprec:                       1e-03    <-- Symmetry precision for spglib in Angström.
workdir:                  relaxation    
mask:                   [1,1,1,1,1,1]   <-- Mask out parts of the stress tensor (xx, yy, zz, yz, xz, xy).

[relaxation.kwargs]
maxstep:                       0.2      <-- Maximum displacement in Angström.
logfile:                       relaxation.log
restart:                       bfgs.restart
alpha: 25                               <-- Initial hessian factor. The value 25 is similar to the one chosen in FHI-aims.
```


## Modifying the phonopy.in file

If you perform a phonon calculation as task, the script produces a template `phonopy.in` file for FHI-vibes. Please work through the [FHI-vibes Phonon Tutorial](https://vibes-developers.gitlab.io/vibes/Tutorial/2_phonopy/) to understand the function of all keywords. Many of the blocks in are similar to the `relaxation.in`.

```vim
[calculator]
name:                          aims

[calculator.parameters]
xc:                            pbe
many_body_dispersion_nl:
tier: 1
sc_accuracy_rho: 1e-6                   <-- Phonons require very accurately converged geometries. There might be cases where you have to decrease this value to 1e-7.
sc_iter_limit: 100

[calculator.kpoints]
density: 5.00                           <-- Choose the k-point density over specifing a k-grid, because this value will stay the same for all supercell sizes.

[calculator.basissets]
default:                       tight    <-- Accurate phonos require tight numerical settings, in some cases maybe even additional modifiers.

[calculator.socketio]
port: auto

[phonopy]
supercell_matrix:          [2, 2, 2]    <-- Check out the utility `vibes utils make-supercell` to find good supercell sizes.
displacement:                  0.01     <-- Larger displacements mean larger forces and less numerical noise, but sometimes worse SCF convergence. Choose 0.01 or 0.005.
is_diagonal:                   True     
is_plusminus:                  auto
symprec:                       1e-03
q_mesh:                 [45, 45, 45]    <-- This option is mainly for plotting.
```