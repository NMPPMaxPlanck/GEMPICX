source /etc/profile
module purge

if [ x"$CLUSTER" == x"RAVEN" ]; then
    module load cuda/12.2
    module load gcc/12
    module load openmpi/5.0 openmpi_gpu/5.0
    module load mkl/2024.0
    module load clang/16
elif [ x"$CLUSTER" == x"VIPER" ]; then
    module load intel/2024.0
    module load openmpi/4.1
    module load mkl/2024.0
    module load clang/17
else
    module load intel/21.3.0
    module load cuda/11.4
    module load openmpi/4.1
    module load mkl/2021.3
    module load gcc/10
    module load clang/11
fi

module load anaconda/3/2023.03
module load git
module load cmake/3.24
module load ninja
module load numdiff
module list