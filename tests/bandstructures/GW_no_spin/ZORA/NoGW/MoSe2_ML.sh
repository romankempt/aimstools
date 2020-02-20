#! /bin/sh
#PBS -N MoSe2_ML
#PBS -l walltime=120:00:00,mem=128GB,nodes=4:ppn=20
module ()
   eval `/usr/bin/modulecmd bash $*`

module load aims

export OMP_NUM_THREADS=1
cd $PBS_O_WORKDIR

mpirun -np 80 bash -c "ulimit -s unlimited && aims.171221_1.scalapack.mpi.x" < /dev/null > MoSe2_ML.out
