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

module purge
module load intel/19.1.1
module load impi/2019.7
module load mkl/2019.5

# Run the program:
srun ./vlasov_maxwell LHW_test.input  > test.out
