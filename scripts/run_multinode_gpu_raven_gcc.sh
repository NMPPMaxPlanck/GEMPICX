#!/bin/bash -l
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
#SBATCH -D ./
#SBATCH -J GEMPIC
#SBATCH --nodes=2            # Request 1 or more node(s)
#SBATCH --tasks-per-node=4   # Launch 1 task per GPU
#SBATCH --cpus-per-task=18   #   with 18 CPU cores each.
#SBATCH --constraint=gpu     # Request
#SBATCH --gres=gpu:4         #   4 GPUs per node.
#SBATCH --mail-type=none
#SBATCH --mail-user=none
#SBATCH --time=00:03:59

# load modules consistently with `compile_gpu_raven-gcc.sh`
module purge
module load gcc/10
module load cuda/11.2
module load impi/2021.4

srun ./vlasov_maxwell PIC_params_Weibel.input >Weibel.out

