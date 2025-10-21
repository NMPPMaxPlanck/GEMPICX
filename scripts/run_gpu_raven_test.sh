#!/bin/bash -l
#
# Slurm job script for a brief gpu test run on Raven
# using one A100 GPU and an OpenMPI-based build of GEMPIC.
# Job submission:  sbatch --wait run_gpu_raven_test.sh
# **Before submission**: you must update the REPOSITORY and DIMENSION variables
# given below.
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

#######################################
# FIXME: please adapt to your own setup
export REPOSITORY=${HOME}/repos/sonnendruecker/gempic
export DIMENSION="3D"
#######################################

source ${REPOSITORY}/scripts/mpcdf_modules.inc

# put TMPDIR onto a RAMDISK for fast builds and cleanup, moreover the
# default /tmp would be too small to hold all temporary files
export TMPDIR=$JOB_SHMTMPDIR
# keep the default build dir relative to the source for the moment
export BUILD_DIR=${REPOSITORY}/build/mpcdf-raven-${DIMENSION}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Run the program:
srun ${BUILD_DIR}/Testing/GtestTests
cd ${BUILD_DIR}
ctest --output-on-failure -L non-Google

