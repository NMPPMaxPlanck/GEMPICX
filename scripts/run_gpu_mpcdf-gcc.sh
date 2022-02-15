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
#SBATCH --mail-user=sonnen@ipp.mpg.de
#
# Wall clock limit:
#SBATCH --time=00:05:00

module purge
module load gcc/10 
module load openmpi
module load cuda/11.2

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
# Run the program:
srun ./test_maxwell_yee > test.out
