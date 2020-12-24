# Converging the k-point density

The *k-point density* is arguably the most important convergence parameter in a solid state calculation, but also one of the most problematic ones. The convergence of the property of interest (total energy, band gap, absorption spectrum) with respect to the k-point density is neither variational nor necessarily monotonous.

This guide presents the steps how to use the automated workflow `KPointConvergence` to facilitate that task.

## Calling the KPointConvergence workflow

You can call the `KPointConvergence` class from python:

```python
from aimstools.workflows import KPointConvergence as KPC
kpc = KPC(geometryfile="geometry.in")
```

Or through the command-line utility `aims_workflow`:

```bash
aims_workflow converge_kpoints geometry.in
```

This workflow has two different modes, called `preparation` and `evaluation`. If you specify a geometry file as input and the result directory `aimstools_kpoint_convergence/` does not exist, the mode will be automatically set to `preparation` and all the files needed for converging the k-grid will be set up. Otherwise, the results from these files will be evaluated.

If there already exists a `control.in` in the current working directory, this one will be used for all calculations.

## Analyzing the Results

When all calculations for the k-point convergence have finished, you can simply call the utility `aims_workflow converge_kpoints /path/to/results/ -i` again, where `-i` specifies the interactive plotting mode.

This should log information similar to this to your console:
```bash
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ k-grid   ┃ k-point density  ┃ total energy  ┃ band gap  ┃ number of SCF cycles ┃ converged ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ 2x2x2    │ 1.01             │ -7900.073544  │ 0.67      │ 12                   │ True      │
│ 4x4x4    │ 2.01             │ -7901.237159  │ 0.77      │ 12                   │ True      │
│ 6x6x6    │ 3.02             │ -7901.317293  │ 0.78      │ 12                   │ True      │
│ 8x8x8    │ 4.02             │ -7901.327057  │ 0.67      │ 12                   │ True      │
│ 10x10x10 │ 5.03             │ -7901.328599  │ 0.64      │ 12                   │ True      │
│ 12x12x12 │ 6.03             │ -7901.328883  │ 0.64      │ 12                   │ True      │
│ 14x14x14 │ 7.04             │ -7901.328942  │ 0.64      │ 12                   │ True      │
│ 16x16x16 │ 8.04             │ -7901.328954  │ 0.64      │ 12                   │ True      │
│ 18x18x18 │ 9.05             │ -7901.328957  │ 0.65      │ 12                   │ True      │
│ 20x20x20 │ 10.05            │ -7901.328958  │ 0.64      │ 12                   │ True      │
│ 22x22x22 │ 11.06            │ -7901.328958  │ 0.64      │ 12                   │ True      │
└──────────┴──────────────────┴───────────────┴───────────┴──────────────────────┴───────────┘
INFO     The k-kgrid is converged within  1.0E-04 eV/atom for a grid of 12x12x12 after 12 SCF cycles.
INFO     The k-kgrid is converged within  1.0E-05 eV/atom for a grid of 16x16x16 after 12 SCF cycles.
INFO     The k-kgrid is converged within  1.0E-06 eV/atom for a grid of 18x18x18 after 12 SCF cycles.
```

The table collects the relevant data from the finished calculations. The values of the `k-grid` are given for different accuracy thresholds. However, these should be taken with a grain of salt, since the property of interest might oscillate even on very dense `k-grids`.

| threshold | usage |
|---|---|
| 1.0E-04 eV/atom | Sparse k-grid, e.g., for initial geometry optimizations. |
| 1.0E-05 eV/atom | Relatively dense k-grid for production calculations. |
| 1.0E-06 eV/atom | Very dense k-grid, typically needed for metallic systems and optical spectra. |