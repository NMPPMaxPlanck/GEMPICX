#!/bin/bash -l
### Script for post-processing data files, collapsing them into fewer files for easier data I/O
# Standard output and error:
#SBATCH -o ./%x.out.%j  #(%x jobname %j jobid)
#SBATCH -e ./%x.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J gempic_post_proc
#
#SBATCH --nodes=1 # Request 1 (or more) node(s)
#SBATCH --ntasks-per-node=72
#
#SBATCH --mail-type=none
#SBATCH --mail-user=@ipp.mpg.de
#
# Wall clock limit:
#SBATCH --time=0:30:00

module purge
module load gcc/11
module load openmpi/4
module load anaconda/3/2023.03
module load mpi4py/3.1.5

# Run the program:
srun python3 ~/codes/gempic/Examples/SupplementaryScripts/CreateSpaceTimeArrays.py ./FullDiagnostics/plt_field  rho
