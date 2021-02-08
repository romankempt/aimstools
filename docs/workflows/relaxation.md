# Geometry Optimizations

This guide just presents the typical steps for a geometry optimization or relaxation in FHI-aims and discusses some recommended settings.
Please keep in mind that some of these recommendations do not work for every system and need careful testing. Also, consider different sources, such as the FHI-vibes tutorials, the FHI-aims guidelines and the FHI-aims manual.

## Initial optimization

Set up an initial optimization with `aims_prepare geometry.xyz --basis light --tier 1 --task GO -k 3`.
This prepares the files for FHI-vibes with light numerical settings and a tier 1 basis set. The k-point density is a value you need to converge separately of the relaxation procedure.

Additionally, you should consider changing the following parameters:

| parameter | values |
|---|---|
| `sc_accuracy_rho` | Depending on the number of atoms, you can choose this value somewhere between `1e-3` to `1e-6`. |
| `fmax` | For an initial relaxation, choose a higher value like `1e-2` or even `5e-2` eV/Angström. |
| `fix_symmetry` | If you fix the symmetry, you introduce a bias to the structure of your system. If you investigate new systems, do not fix the symmetry. |
| `mask` | This keyword is extremely useful for 2D materials. Set it to [1,1,0,0,0,1] to remove all forces on the unit cell in the `z`-direction. |


<div class="warning">
The <mark>mask</mark> keyword is a constraint that removes forces from the system.
This can make your optimization significantly slower, especially for bulk structures.
Additionally, this means that the <mark>residual force</mark> in the <mark>relaxation.log</mark> does not correspond to the actual residual force. 
Try to avoid masks if you can (except for 2D materials). If necessary, the mask can be used to fix unit cell angles, e.g., by setting it to <mark>[1,1,1,0,0,0]</mark>.
<b>Do not use the mask keyword together with the fix_symmetry keyword.</b> This corresponds to doubling constraints, making the relaxation artificial.
</div>

Let the initial optimization run for about 10-50 steps, depending on your system. If the optimization does not converge, that does not necessarily mean that something is wrong with your system. This might just be due to the `light` settings used here.

## Full optimization

Symmetrize and refine your structure either by using the `--standardize` option of `aims_prepare` or via the python interface.
Additionally, change the basis sets to `tight tier 1`.

<div class="tip">
The recommended setting for production calculations from FHI-aims is <b>tight tier 2</b>. However, the second tier mainly adds diffuse functions to the basis set. These are not as relevant for dense periodic systems as for molecules, except for cases involving anions and some organic groups. You need to test the difference for your system, but I've typically been fine by using a tier 1 basis set, which drastically reduces calculation costs especially when hybrid functionals are used.
</div>

Change the following values:

| parameter | values |
| --- | --- |
| `sc_accuracy_rho` | Stick to `1e-6`. |
| `fmax` | Choose `5e-2` or `1e-3` if possible. |
| `fix_symmetry` | When you have refined the structure, you may fix the symmetry. |

If your structure does not converge for a stricter criterion of `fmax`, this might be an indication that you have to choose a denser `k_grid` or even tighter numerical settings, such as `sc_accuracy_rho 1e-7`. If your system is very flexible (e.g., rotating organic groups), then it might not be possible to set such a strict convergence criterion.

## Converging geometries for phonons

After you finished the second step, you might run phonon calculations. In case you encounter small imaginary frequencies, especially small pockets around the Gamma-point, this *typically* just means you need to converge your geometry a little tighter. Then, you might consider choosing:

| parameter | values |
| --- | --- |
| `sc_accuracy_rho` | `1e-7` |
| `sc_accuracy_forces` | `1e-6` |
| `fmax` | `5e-4` |
| `maxstep` | `0.05` |

Additionally, denser k-grids may help or some more structural refinement.

