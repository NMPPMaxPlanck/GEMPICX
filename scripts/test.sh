#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./tjob.out.%j
#SBATCH -e ./tjob.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J gempic
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#
# Memory usage [MB] of the job is required, 2200 MB per task:
#SBATCH --mem=17600
#
#SBATCH --mail-type=none
#SBATCH --mail-user=sonnen@ipp.mpg.de
#
# Wall clock limit:
#SBATCH --time=00:30:00

RUN_DIR = ${HOME}/gempic_runs
source mpcdf_modules.inc

# Run the program:
#srun ./vlasov_maxwell PIC_params_Weibel.input > Weibel.out
source /etc/profile
module purge
if [ x"$CLUSTER" == x"COBRA"  ]; then
module load intel/19.1.1
module load impi/2019.7
module load mkl/2019.5
fi
if [ x"$CLUSTER" == x"RAVEN" ]; then
module load intel/21.4.0
module load impi/2021.4
module load mkl/2021.2
fi
module load cmake/3.18
module load gcc
module list
