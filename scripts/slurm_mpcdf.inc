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
#SBATCH --ntasks-per-node=72
#
#SBATCH --mail-type=none
#SBATCH --mail-user=sonnen@ipp.mpg.de
#
# Wall clock limit:
#SBATCH --time=20:00:00

