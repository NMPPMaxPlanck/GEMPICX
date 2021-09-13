SOURCE_DIRECTORY=`pwd`/../
SOURCE_DIRECTORY=`readlink -f $SOURCE_DIRECTORY`
AMREX_DIRECTORY=`pwd`/../../amrex
AMREX_DIRECTORY=`readlink -f $AMREX_DIRECTORY`

echo $SOURCE_DIRECTORY
echo $AMREX_DIRECTORY

module purge
module load intel/19.1.2
module load impi/2019.8
module load mkl/2021.2
module load cmake/3.15
module load cuda/10.2

BUILD_DIR=$HOME/gempic_gpu_obj

mkdir -p $BUILD_DIR
cd $BUILD_DIR

mkdir -p amrex
cd amrex

cmake -D AMReX_GPU_BACKEND=CUDA -D AMReX_SPACEDIM=3 -D AMReX_PARTICLES=ON --expt-extended-lambda $AMREX_DIRECTORY

make install -j 16

cd ..
rm -rf gempic
mkdir -p gempic
cd gempic

cmake -D AMReX_ROOT=$AMREX_DIRECTORY/installdir -D USE_CUDA=ON -D CUDA_HOST_COMPILER=mpiicc -D CMAKE_BUILD_TYPE=Release $SOURCE_DIRECTORY
gmake VERBOSE=1 -j 16
