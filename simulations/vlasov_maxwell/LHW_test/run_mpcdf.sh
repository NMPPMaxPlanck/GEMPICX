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
#SBATCH --ntasks-per-node=32
#
# Memory usage [MB] of the job is required, 2200 MB per task:
#SBATCH --mem=17600
#
#SBATCH --mail-type=end
#SBATCH --mail-user=dliu@ipp.mpg.de
#
# Wall clock limit:
#SBATCH --time=00:05:00

module load intel/19.0.4
module load impi/2019.4
module load mkl/2019.4
export LD_LIBRARY_PATH=/lib64:/mpcdf/soft/SLE_12/packages/x86_64/intel_parallel_studio/2019.4/compilers_and_libraries_2019.4.243/linux/compiler/lib/intel64_lin 
# Run the program:
srun ./vlasov_maxwell LHW_test.input  > test.out
