#!/bin/bash -l
#
# Slurm job script for a brief gpu test run
# using one A100 GPU and an OpenMPI-based build of GEMPIC.
# Job submission:  sbatch --wait run_gpu_raven_test.sh
#
#SBATCH -o ./testSim.out
#SBATCH -D ./
#SBATCH -J simulation_gpu
#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de

#SBATCH --nodes=1 # max 1 node for gpudev
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240000
#SBATCH --gres=gpu:1
#SBATCH --partition=apudev
#SBATCH --time=00:15:00 # max time for gpudev is 15 min

# exit immediately at any error
set -e

#######################################
# FIXME: please adapt to your own setup
export REPOSITORY=${HOME}/gempic
export DIMENSION="2D"
#######################################

source ${REPOSITORY}/scripts/mpcdf_modules.inc

# put TMPDIR onto a RAMDISK for fast builds and cleanup, moreover the
# default /tmp would be too small to hold all temporary files
export TMPDIR=$JOB_SHMTMPDIR
# keep the default build dir relative to the source for the moment
export BUILD_DIR=${REPOSITORY}/build/mpcdf-viper-${DIMENSION}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Run the program:
#srun $BUILD_DIR/Testing/GtestTests

# works, 90 GB GPU mem occupancy in debug mode
#export EXECUTABLE=$BUILD_DIR/Examples/Electrostatic/electrostatic
#export INPUT_FILE=ITG.input

# testing
export EXECUTABLE=$BUILD_DIR/Examples/VlasovMaxwell/vlasovmaxwell
export INPUT_FILE=astro.input

# simple example run
# srun ${EXECUTABLE} ${INPUT_FILE} > job.out

### Profiling runs

## basic timeline information, pftrace files can be loaded in google chrome at address ui.perfetto.dev
## useful to get expensive kernels
#srun \
#    rocprofv3 \
#    --stats \
#    --runtime-trace \
#    --output-format pftrace \
#    -S \
#    -- \
#    ${EXECUTABLE} ${INPUT_FILE} > job.out


## roofline analysis of specific kernel names (see relevant kernels from runtime-trace job)
## IMPORTANT: only works on full node, otherwise counter collection may fail
##            (bus error something)
##            In other words: script needs "--gres=gpu:2" above
#mpirun -np 1 \
#    rocprof-compute \
#    profile -n rooflines_PDF \
#    --device 0 \
#    --roof-only \
#    --kernel-names \
#    -k _ZN5amrex13launch_globalILi256EZNS_11ParallelForILi256ElZ4mainEUllE_vEENSt9enable_ifIXsr19MaybeDeviceRunnableIT1_EE5valueEvE4typeERKNS_3Gpu10KernelInfoET0_RKS4_EUlvE_EEvSB_ \
#    -- \
#    ${EXECUTABLE} ${INPUT_FILE} > job.out
