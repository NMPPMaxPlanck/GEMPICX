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
    #  see  https://docs.mpcdf.mpg.de/doc/computing/viper-gpu-user-guide.html#recommended-compiler-and-mpi-software-stack-on-viper-gpu
    # problem: The $CLUSTER environment variable is the same for viper-cpu andi
    #          viper-gpu (as of 2025-05-20).
    #          We need different modules on the different clusters.
    #          So we test the hostname.
    if [ x"$HOSTNAME" == x"viper12" ]; then
        # viper-GPU
        module load amd-llvm/6.1
        module load rocm/6.4
        module load openmpi/5.0
        #module load openmpi_gpu/5.0 # FIXME: this is supposed to be GPU-aware MPI, needs to be tested.
        module load mkl/2025.1
        module load clang/18
    else
        # viper-CPU
        module load intel/2025.1
        module load impi/2021.15
        module load mkl/2025.1
        module load clang/18
    fi
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
module load python-waterboa/2024.06
module load cmake/3.30
module load ninja
module list
