#!/bin/bash
#SBATCH --time=5:00:00     			# walltime in h
#SBATCH --nodes=1             		# number of nodes
#SBATCH --ntasks-per-node=24     	# cpus per node
#SBATCH --exclusive                 # assert that job takes full nodes
#SBATCH --mem-per-cpu=2583MB    	# memory per node per cpu (1972MB on romeo)
#SBATCH -J hBN_BS_DOS_fatBS                   # job name  
#SBATCH --error=slurm.err           	# stdout
#SBATCH --output=slurm.out          	# stderr
##SBATCH --partition=haswell64      	# partition name, can also be gpu2 or romeo
##SBATCH --gres=gpu:4               	# using 4 gpus per node

# when using gpus, assert that use_gpu is in control.in !
# gpu1 partition will not be supported by the end of the year

module use /home/kempt/Roman_AIMS_env/modules
module load aims_env

echo "slurm job ID: $SLURM_JOB_ID"

# srun_aims is a bash script given by aims_env that executes the AIMS binary and sets additional environment variables
source srun_aims > aims.out
            