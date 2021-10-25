SOURCE_DIRECTORY=`pwd`/../
SOURCE_DIRECTORY=`readlink -f $SOURCE_DIRECTORY`
AMREX_DIRECTORY=`pwd`/../../amrex
AMREX_DIRECTORY=`readlink -f $AMREX_DIRECTORY`

echo $SOURCE_DIRECTORY
echo $AMREX_DIRECTORY

#cp $SOURCE_DIRECTORY/third_party/AMReX_MLNodeLap_Gempic_3D_K.H $AMREX_DIRECTORY/Src/LinearSolvers/MLMG/AMReX_MLNodeLap_3D_K.H
module purge
module load intel/19.1.2
module load impi/2019.8
module load mkl/2021.2
module load cmake/3.18
module load cuda/11.2

#module add cudatoolkit
#module add daint-gpu
#module add CMake/3.14.5
#module del PrgEnv-cray
#module add PrgEnv-gnu
#module switch gcc/9.3.0

BUILD_DIR=$HOME/gempic_gpu_obj3

#rmdir -r $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR

#rm -rf amrex
mkdir -p amrex
mkdir -p amrex_install
cd amrex

cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_FLAGS="-std=c++17" -D AMReX_GPU_BACKEND=CUDA -D AMReX_CUDA_ARCH=8.0 -D AMReX_SPACEDIM=3 -D AMReX_PARTICLES=ON -D CMAKE_CXX_COMPILER=mpiicpc -D CMAKE_C_COMPILER=mpiicc -D CUDA_HOST_COMPILER=mpiicpc -D CMAKE_CPP_COMPILER=mpiicpc -D CMAKE_LINKER=mpiicpc -D CMAKE_CUDA_FLAGS=-ccbin=mpiicpc -D CMAKE_INSTALL_PREFIX=../amrex_install --expt-extended-lambda $AMREX_DIRECTORY

make install -j 16

cd ..
rm -rf gempic
mkdir -p gempic
cd gempic

CC=mpiicpc cmake -D AMReX_DIR=$HOME/gempic_gpu_obj3/amrex_install -D CMAKE_CXX_FLAGS="-std=c++17" -D CMAKE_CXX_COMPILER=mpiicpc -D CMAKE_C_COMPILER=mpiicc -D USE_CUDA=ON -D CUDA_HOST_COMPILER=mpiicpc -D CMAKE_CPP_COMPILER=mpiicpc -D CMAKE_CUDA_FLAGS=-ccbin=mpiicpc -D CMAKE_MPI_CUDA_INCLUDE_DIRS='/mpcdf/soft/SLE_15/packages/x86_64/intel_parallel_studio/2020.2/compilers_and_libraries_2020.2.254/linux/mpi/intel64/include/'  -D CMAKE_CUDA_ARCHITECTURES=80 -D CMAKE_BUILD_TYPE=Release ~/gempic -D CMAKE_EXE_LINKER_FLAGS=-lnvToolsExt

gmake -j 16
