#!/bin/bash -l
#
# Slurm job script for a CPU run on RAVEN or VIPER
# Job submission:  sbatch --wait run_cpu_mpcdf.sh
#
#SBATCH -o ./GEMPICXcpuSim.out
#SBATCH -D ./
#SBATCH -J GEMPICXsimulation_cpu

#SBATCH --nodes=1 #
#SVATCG --ntasks-per-node=72
#SBATCH --mem=125000
#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de
#SBATCH --time=10:00:00 # 

# exit immediately at any error
set -e
source ./mpcdf_modules.inc

# put TMPDIR onto a RAMDISK for fast builds and cleanup, moreover the
# default /tmp would be too small to hold all temporary files
export TMPDIR=$JOB_SHMTMPDIR
# keep the default build dir relative to the source for the moment
export BUILD_DIR=$(pwd)/../build/cpu-release-3D

# Run the program:
srun $BUILD_DIR/Examples/ElectrostaticSimulation bernstein.input > job.out