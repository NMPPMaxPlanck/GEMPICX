#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./tjob.out.%j
#SBATCH -e ./tjob.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J gempic
#
## Plain MPI job using GPUs
#SBATCH --nodes=1 # Request 1 (or more) node(s)
#SBATCH --constraint="gpu"
#SBATCH --ntasks-per-node=72   # Launch 72 tasks per node
#SBATCH --gres=gpu:a100:4     # Request all 4 GPUs of each node
#SBATCH --nvmps               # Launch NVIDIA MPS to enable concurrent access to the GPUs from multiple processes efficiently
#
#SBATCH --mail-type=none
#SBATCH --mail-user=@ipp.mpg.de
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