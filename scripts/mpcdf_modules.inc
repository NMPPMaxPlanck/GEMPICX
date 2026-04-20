# Common module include file for RAVEN/VIPER work
# source scripts/mpcdf_modules.inc

# cleanup
module purge

################################################################################
# helpful functions
load_viper_gpu () {
    # viper-GPU
    # amd-llvm contains the latest compilers from AMD
    # rocm contains hipBLAS and similar.
    # it is important for amd-llvm to be loaded **after** rocm, such that
    # the amd-llvm compilers are used.
    module load rocm/7.0
    module load amd-llvm/7.1
    module load openmpi/5.0 openmpi_gpu/5.0
    module load mkl/2025.2
    module load gcc/15
    export GEMPIC_VIPER_LOADED="true"
    # see https://docs.mpcdf.mpg.de/doc/computing/software/compilers_languages.html#compiling-and-linking-against-a-more-recent-c-standard-library
    export LDFLAGS="$LDFLAGS -L${GCC_HOME}/lib64 -Wl,-rpath,${GCC_HOME}/lib64"
}
string_matches() {
    big_string="$1"
    string_to_search_for="$2"
    echo "$big_string" | grep -q "$string_to_search_for"
}
################################################################################

################################################################################
# generic modules
module load git
module load cmake
module load ninja
################################################################################

if string_matches ${CLUSTER} "RAVEN"; then
    module load gcc/15
    module load cuda/13.0
    module load openmpi/5.0 openmpi_gpu/5.0
    module load mkl/2024.0
    module load clang/18
elif string_matches ${CLUSTER} "VIPER"; then
    #  see  https://docs.mpcdf.mpg.de/doc/computing/viper-gpu-user-guide.html#recommended-compiler-and-mpi-software-stack-on-viper-gpu
    # problem: The $CLUSTER environment variable is the same for viper-cpu and
    #          viper-gpu (as of 2025-05-20).
    #          We need different modules on the different clusters.
    #          So we test the hostname.
    # "viper[11-13]" --- viper-gpu login node
    # "vipa*"   --- viper-gpu compute node
    export GEMPIC_VIPER_LOADED="false"
    if string_matches ${HOSTNAME} "viper1[1-3]"; then
        load_viper_gpu
    fi
    if string_matches ${HOSTNAME} "vipa"; then
        load_viper_gpu
    fi
    if string_matches ${GEMPIC_VIPER_LOADED} "false"; then
        # viper-CPU
        module load intel/2025.1
        module load impi/2021.15
        module load mkl/2025.1
        module load clang/18
        module load gcc/15
        export GEMPIC_VIPER_LOADED="true"
    fi
else
    module load intel/21.3.0
    module load cuda/11.4
    module load impi/2021.3
    module load mkl/2021.3
    module load gcc/10
    module load clang/11
fi

################################################################################
# modules that may depend on compiler etc
module load fftw-mpi
module load python-waterboa/2024.06
################################################################################

module list
