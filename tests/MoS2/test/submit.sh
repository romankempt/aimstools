#!/bin/bash
#SBATCH --time=23:00:00 		# walltime in h
#SBATCH --nodes=1 			# number of nodes
#SBATCH --ntasks=24 			# number of cpus
#SBATCH --ntasks-per-node=24 	 	 # cpus per node
#SBATCH --exclusive
#SBATCH --mem-per-cpu=2583MB 		# memory per node per cpu
#SBATCH -J Mo2S4_GO_BS 			# job name
#SBATCH --error=slurm.err 		# stdout
#SBATCH --output=slurm.out 		# stderr

module use /home/kempt/Roman_AIMS_env/modules
module load aims_env

COMPUTE_DIR=aims_$SLURM_JOB_ID
ws_allocate -F ssd $COMPUTE_DIR 7
export AIMS_SCRDIR=/ssd/ws/$USER-$COMPUTE_DIR

srun $AIMS_EXECUTABLE > aims.out
            