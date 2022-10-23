#!/bin/bash -l
#
# Slurm job script for a continuous integration run on Raven,
# using one A100 GPU and an OpenMPI-based build of GEMPIC.
# Job submission:  sbatch --wait ci_raven_gpu_jobscript.sh
#
#SBATCH -o ./ci_job.out
#SBATCH -D ./
#SBATCH -J ci_gpu
#SBATCH --nodes=1 
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
#SBATCH --time=0:60:00

# exit immediately at any error
set -e

# put TMPDIR onto a RAMDISK for fast builds and cleanup, moreover the
# default /tmp would be too small to hold all temporary files
export TMPDIR=$JOB_SHMTMPDIR
# keep the default build dir relative to the source for the moment,
# because the ctests seem to require this for some reason
export BUILD_DIR=$(pwd)/../build

# set number of OMP threads (4 processes \times 4 threads in use during tests)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# CI phase 1 -- build
time -p ./ci_raven_gpu_build_gcc.sh

# CI phase 2 -- run the program
time -p srun ../build/simulations/vlasov_maxwell/vlasov_maxwell ${SOURCE_DIRECTORY}/simulations/vlasov_maxwell/PIC_params_Landau04.input
python3 ${SOURCE_DIRECTORY}/post_processing/Landau04err.py