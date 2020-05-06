SOURCE_DIRECTORY=`pwd`/../
SOURCE_DIRECTORY=`readlink -f $SOURCE_DIRECTORY`
AMREX_DIRECTORY=`pwd`/../../amrex
AMREX_DIRECTORY=`readlink -f $AMREX_DIRECTORY`

echo $SOURCE_DIRECTORY
echo $AMREX_DIRECTORY

module purge
module load intel/18.0.5
module load impi/2018.4
module load mkl/2018.4
module load cmake/3.15

BUILD_DIR=/ptmp/$USER/gempic_obj

mkdir -p $BUILD_DIR
cd $BUILD_DIR

mkdir -p amrex
cd amrex

cmake -DDIM=3 -D ENABLE_PARTICLES=ON $AMREX_DIRECTORY

make install

cd ..
mkdir -p gempic
cd gempic
cmake -D AMReX_ROOT=$AMREX_DIRECTORY/installdir -D CMAKE_C_COMPILER=mpiicc -D CMAKE_CXX_COMPILER=mpiicpc -D CMAKE_CXX_FLAGS="-std=c++14" $SOURCE_DIRECTORY
make
