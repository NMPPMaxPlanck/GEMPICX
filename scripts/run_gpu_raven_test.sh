#!/bin/bash -l
#
# Slurm job script for a brief gpu test run on Raven
# using one A100 GPU and an OpenMPI-based build of GEMPIC.
# Job submission:  sbatch --wait run_gpu_raven_test.sh
#
#SBATCH -o ./testSim.out
#SBATCH -D ./
#SBATCH -J simulation_gpu

#SBATCH --partition=gpudev
#SBATCH --nodes=1             # max 1 node for gpudev
#SBATCH --constraint="gpu"
#SBATCH --ntasks-per-node=1   # Launch 1 task per GPU
#SBATCH --cpus-per-task=1     # No OMP, just one CPU core per task/GPU
#SBATCH --mem=125000
#SBATCH --gres=gpu:a100:1     # Request a single GPU

#SBATCH --mail-type=none
#SBATCH --mail-user=userid@ipp.mpg.de
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

