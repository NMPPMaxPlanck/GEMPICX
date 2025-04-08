#!/bin/bash -l
#
# Slurm job script for a brief gpu test run
# using one A100 GPU and an OpenMPI-based build of GEMPIC.
# Job submission:  sbatch --wait run_gpu_raven_test.sh
#
#SBATCH -o ./testSim.out
#SBATCH -D ./
#SBATCH -J simulation_gpu

#SBATCH --partition=gpudev
#SBATCH --nodes=1 # max 1 node for gpudev
#SBATCH --constraint="gpu"
#SVATCG --ntasks-per-node=72
#SBATCH --mem=125000
#SBATCH --gres=gpu:a100:4
#SBATCH --nvmps # Launch NVIDIA MPS to enable concurrent access to the GPUs from multiple processes efficiently
#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de
#SBATCH --time=00:15:00 # max time for gpudev is 15 min

# exit immediately at any error
set -e
source ./mpcdf_modules.inc

# put TMPDIR onto a RAMDISK for fast builds and cleanup, moreover the
# default /tmp would be too small to hold all temporary files
export TMPDIR=$JOB_SHMTMPDIR
# keep the default build dir relative to the source for the moment
export BUILD_DIR=$(pwd)/../build/mpcdf-gpu-3D

# Run the program:
srun $BUILD_DIR/Testing/GtestTests

