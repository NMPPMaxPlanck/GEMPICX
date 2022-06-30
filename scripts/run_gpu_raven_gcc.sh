#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./tjob.out.%j
#SBATCH -e ./tjob.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J gempic
#
#SBATCH --nodes=1 # Request 1 (or more) node(s)
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=72 # Launch 72 tasks per node
#SBATCH --nvmps
#
#SBATCH --mail-type=end
#SBATCH --mail-user=sonnen@ipp.mpg.de
#
# Wall clock limit:
#SBATCH --time=24:00:00

module purge
module load gcc/11 
module load cuda/11.4
module load openmpi_gpu/4

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
# Run the program:
srun ./vlasov_maxwell PIC_params_Weibel.input > Weibel.out
