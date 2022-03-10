#!/bin/bash

set -e

source ci_raven_gpu_build_gcc.inc

BUILD_DIR="${BUILD_DIR:-../build}"

# enable `mpirun` for OpenMPI locally by disabling the Slurm variables
unset "${!SLURM_@}"

cd ${BUILD_DIR}/gempic
ctest --output-on-failure

