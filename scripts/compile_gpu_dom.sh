SOURCE_DIRECTORY=`pwd`/../
SOURCE_DIRECTORY=`readlink -f $SOURCE_DIRECTORY`
AMREX_DIRECTORY=`pwd`/../../amrex
AMREX_DIRECTORY=`readlink -f $AMREX_DIRECTORY`

echo $SOURCE_DIRECTORY
echo $AMREX_DIRECTORY

#cp $SOURCE_DIRECTORY/third_party/AMReX_MLNodeLap_Gempic_3D_K.H $AMREX_DIRECTORY/Src/LinearSolvers/MLMG/AMReX_MLNodeLap_3D_K.H
#module purge
#module load intel/19.1.2
#module load impi/2019.8
#module load mkl/2021.2
#module load cmake/3.15
#module load cuda/10.2

#module add cudatoolkit
#module add daint-gpu
#module add CMake/3.14.5
#module del PrgEnv-cray
#module add PrgEnv-gnu
#module switch gcc/9.3.0
module swap PrgEnv-cray PrgEnv-gnu
module load cdt-cuda/21.05
module load craype-accel-nvidia60
module swap cudatoolkit cudatoolkit/11.2.0_3.39-2.1__gf93aa1c
module load daint-gpu
module load CMake/3.14.5
module use /apps/daint/UES/eurohack/modules/all
module load nsys/2021-3
module use /apps/daint/UES/eurohack/modules/all
module load numdiff/5.9.0
module list

BUILD_DIR=$SCRATCH/gempic_gpu_obj2

#rmdir -r $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR

mkdir -p amrex
cd amrex

CXX=CC cmake -D AMReX_GPU_BACKEND=CUDA -D AMReX_SPACEDIM=3 -D AMReX_PARTICLES=ON -D AMReX_TINY_PROFILE=ON -D CUDA_HOST_COMPILER=/opt/cray/pe/craype/2.7.3/bin/CC -D CMAKE_CPP_COMPILER=CC -D CMAKE_LINKER=CC -D CMAKE_CUDA_FLAGS=-ccbin=CC --expt-extended-lambda $AMREX_DIRECTORY

make install -j 16

cd ..
rm -rf gempic
mkdir -p gempic
cd gempic

CXX=CC cmake -D AMReX_DIR=$AMREX_DIRECTORY/installdir -D GEMPIC_USE_CUDA=ON -D CUDA_HOST_COMPILER=/opt/cray/pe/craype/2.7.3/bin/CC -D CMAKE_CPP_COMPILER=CC -D CMAKE_CUDA_FLAGS=-ccbin=CC -D MPI_CUDA_INCLUDE_PATH='' -D MPI_CUDA_LIBRARIES='' -D CMAKE_BUILD_TYPE=Debug $SOURCE_DIRECTORY
gmake VERBOSE=1 -j 16
