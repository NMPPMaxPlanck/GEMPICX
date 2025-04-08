# Common module include file for RAVEN/VIPER work
# source scripts/mpcdf_modules.inc
module purge

if [ x"$CLUSTER" == x"RAVEN" ]; then
    module load gcc/13
    module load cuda/12.6
    module load openmpi/5.0 openmpi_gpu/5.0
    module load mkl/2024.0
    module load clang/18
elif [ x"$CLUSTER" == x"VIPER" ]; then
    module load intel/2024.0
    module load impi/2021.11
    module load mkl/2024.0
    module load clang/17
else
    module load intel/21.3.0
    module load cuda/11.4
    module load impi/2021.3
    module load mkl/2021.3
    module load gcc/10
    module load clang/11
fi

module load fftw-mpi
module load git
module load anaconda/3/2023.03
module load cmake/3.30
module load ninja
module list
