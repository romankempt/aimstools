#!/bin/bash
#SBATCH --time=23:00:00 		# walltime in h
#SBATCH --nodes=1 			# number of nodes
#SBATCH --ntasks=24 			# number of cpus
#SBATCH --mem-per-cpu=2583MB 		# memory per node per cpu
#SBATCH -J PtSe2_1T_1L_GO 			# job name
#SBATCH --error=slurm.out 		# stdout
#SBATCH --output=slurm.err 		# stderr

module use /projects/m_chemie/privatemodules/
module add aims/aims_200112

COMPUTE_DIR=aims_$SLURM_JOB_ID
ws_allocate -F ssd $COMPUTE_DIR 7
export AIMS_SCRDIR=/ssd/ws/$USER-$COMPUTE_DIR

export OMP_NUM_THREADS=1
srun aims.200112.scalapack.mpi.x > aims.out
            