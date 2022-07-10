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
# Launch 4 tasks w/ each 18 cores & 1 GPU per node
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=18
#
#SBATCH --mail-type=end
#SBATCH --mail-user=sonnen@ipp.mpg.de
#
# Wall clock limit:
#SBATCH --time=0:30:00

module purge
module load gcc/11 
module load cuda/11.4
module load openmpi/4
module load openmpi_gpu/4

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
# Run the program:
srun ./vlasov_maxwell PIC_params_Weibel.input > Weibel.out
