#!/bin/bash
#SBATCH --time=5:00:00                 # walltime in h
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=128                   # number of cpus
#SBATCH -J BN_bs                 # job name
#SBATCH --error=slurm.out               # stdout
#SBATCH --output=slurm.err              # stderr
#SBATCH --exclusive

echo "slurm job ID: $SLURM_JOB_ID"

srun /path/to/aims/executable > aims.out
