vibes_relaxation_template = """[calculator]
name:                          aims

[calculator.parameters]
xc:                            {xc}
spin:                          {spin}
many_body_dispersion_nl:
tier: {tier}
sc_accuracy_rho: 1e-6
charge_mix_param: 0.2
occupation_type: gaussian 0.01
sc_iter_limit: 100
{calculator_kwargs}

[calculator.kpoints]
density: {kptdensity}

[calculator.basissets]
default:                       {basis}

[calculator.socketio]
port: auto

[relaxation]
driver:                        BFGS
fmax:                          1e-3
unit_cell:                     True
fix_symmetry:                  False
hydrostatic_strain:            False
constant_volume:               False
scalar_pressure:               0.0
decimals:                      12
symprec:                       1e-03
workdir:                       relaxation
mask:                          {mask}

[relaxation.kwargs]
maxstep:                       0.2
logfile:                       relaxation.log
restart:                       bfgs.restart
alpha: 25
"""


vibes_phonopy_template = """[calculator]
name:                          aims

[calculator.parameters]
xc:                            {xc}
spin:                          {spin}
many_body_dispersion_nl:
tier: {tier}
sc_accuracy_rho: 1e-6
charge_mix_param: 0.2
occupation_type: gaussian 0.01
sc_iter_limit: 100
{calculator_kwargs}

[calculator.kpoints]
density: {kptdensity}

[calculator.basissets]
default:                       {basis}

[calculator.socketio]
port: auto

[phonopy]
supercell_matrix:              [2, 2, 2]
displacement:                  0.01
is_diagonal:                   True
is_plusminus:                  auto
symprec:                       1e-03
q_mesh:                        [45, 45, 45]
workdir:                       phonopy"""


aims_slurm_template = """#!/bin/bash
#SBATCH --time=5:00:00                 # walltime in h
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=128                   # number of cpus
#SBATCH -J {jobname}                 # job name
#SBATCH --error=slurm.out               # stdout
#SBATCH --output=slurm.err              # stderr
#SBATCH --exclusive

echo "slurm job ID: $SLURM_JOB_ID"

srun /path/to/aims/executable > aims.out"""

vibes_slurm_template = """#!/bin/bash
#SBATCH --time=5:00:00                 # walltime in h
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=128                   # number of cpus
#SBATCH -J {jobname}                 # job name
#SBATCH --error=slurm.out               # stdout
#SBATCH --output=slurm.err              # stderr
#SBATCH --exclusive

echo "slurm job ID: $SLURM_JOB_ID"

vibes run {task} > log.{task}"""
