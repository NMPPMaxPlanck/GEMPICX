#!/bin/bash

source ci_raven_gpu_build_gcc.inc

# enable `mpirun` for OpenMPI locally by disabling the Slurm variables
unset "${!SLURM_@}"

# set number of OMP threads to a useful value (4 processes \times 4 threads)
export OMP_NUM_THREADS=4

cd ${BUILD_DIR}/gempic
ctest --output-on-failure

