#!/bin/bash -l
#
# Slurm job script for (larger) gpu runs on Raven
# using one or multiple GPU nodes (4 x A100 GPU each).
# Job submission:  sbatch run_gpu_raven.sh
#
# Standard output and error:
#SBATCH -o ./tjob.out.%j
#SBATCH -e ./tjob.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J gempic
#
## Plain MPI job using GPUs
#SBATCH --nodes=1             # Request 1 (or more) node(s)
#SBATCH --constraint="gpu"
#SBATCH --ntasks-per-node=4   # Launch 1 task per GPU
#SBATCH --cpus-per-task=1     # No OMP, just one CPU core per task/GPU
#SBATCH --gres=gpu:a100:4     # Request all 4 GPUs of each node
#
#SBATCH --mail-type=none
#SBATCH --mail-user=userid@ipp.mpg.de
#
# Wall clock limit:
#SBATCH --time=0:30:00

set -e
source mpcdf_modules.inc

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export BUILD_DIR=$(pwd)/../build/mpcdf-gpu-3D
# Run the program normally:
srun $BUILD_DIR/Examples/Electrostatic/electrostatic bernstein.input > job.out
# Run the program with nsight active:
# module load nsight_compute/2024
# module load nsight_systems/2024
## Either:
# nsys profile -o nsight-test --force-overwrite true srun $BUILD_DIR/Examples/VlasovMaxwell/vlasovmaxwell weibel.input > weibel.out
## OR
# srun nsys profile -s cpu --cpuctxsw process-tree -b dwarf -t cuda,nvtx,osrt,mpi --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true $BUILD_DIR/Examples/VlasovMaxwell/vlasovmaxwell weibel.input > weibel.out