#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./tjob.out.%j
#SBATCH -e ./tjob.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J gempic
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#
# Memory usage [MB] of the job is required, 2200 MB per task:
#SBATCH --mem=17600
#
#SBATCH --mail-type=none
#SBATCH --mail-user=kako@ipp.mpg.de
#
# Wall clock limit:
#SBATCH --time=00:05:00

module purge
module load intel/19.1.1
module load impi/2019.7
module load mkl/2019.5
#export LD_LIBRARY_PATH=/lib64:/mpcdf/soft/SLE_12/packages/x86_64/intel_parallel_studio/2019.4/compilers_and_libraries_2019.4.243/linux/compiler/lib/intel64_lin 
# Run the program:
srun ./PIC > test.out
