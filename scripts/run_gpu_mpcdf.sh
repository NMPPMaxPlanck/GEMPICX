#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./tjob.out.%j
#SBATCH -e ./tjob.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J gempic
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#
#SBATCH --mail-type=none
#SBATCH --mail-user=irga@ipp.mpg.de
#
# Wall clock limit:
#SBATCH --time=00:05:00

module purge
module load intel/19.1.2
module load impi/2019.8
module load mkl/2021.2
module load cmake/3.15
module load cuda/10.2

# export LD_LIBRARY_PATH=/lib64:/mpcdf/soft/SLE_12/packages/x86_64/intel_parallel_studio/2019.4/compilers_and_libraries_2019.4.243/linux/compiler/lib/intel64_lin 
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
# Run the program:
srun ./test_maxwell_yee > test.out
