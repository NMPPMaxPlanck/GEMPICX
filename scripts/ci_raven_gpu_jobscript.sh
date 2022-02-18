#!/bin/bash -l
#
# Slurm job script for a continuous integration run on Raven,
# using one A100 GPU and an OpenMPI-based build of GEMPIC.
# Job submission:  sbatch --wait ci_raven_gpu_jobscript.sh
#
#SBATCH -o ./ci_job.out
#SBATCH -D ./
#SBATCH -J ci_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de
#SBATCH --time=00:15:00

# exit immediately at any error
set -e

# put TMPDIR onto a RAMDISK for fast builds and cleanup, moreover the
# default /tmp would be too small to hold all temporary files
export TMPDIR=$JOB_SHMTMPDIR
# keep the default build dir relative to the source for the moment,
# because the ctests seem to require this for some reason
export BUILD_DIR=$(pwd)/../build

# empty the log file
truncate --size=0 ci_job.out

# CI phase 1 -- build
time -p ./ci_raven_gpu_build_gcc.sh

# CI phase 2 -- ctest (uses `srun` internally)
time -p ./ci_raven_gpu_ctest.sh

