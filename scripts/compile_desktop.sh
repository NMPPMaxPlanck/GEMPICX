SOURCE_DIRECTORY=`pwd`/../
SOURCE_DIRECTORY=`readlink -f $SOURCE_DIRECTORY`
AMREX_DIRECTORY=`pwd`/../../amrex
AMREX_DIRECTORY=`readlink -f $AMREX_DIRECTORY`

echo $SOURCE_DIRECTORY
echo $AMREX_DIRECTORY

BUILD_DIR=$HOME/gempic_obj

mkdir -p $BUILD_DIR
cd $BUILD_DIR

mkdir -p amrex
cd amrex

cmake -D AMReX_SPACEDIM=3 -D AMReX_PARTICLES=ON -D CMAKE_BUILD_TYPE=Release $AMREX_DIRECTORY

make install -j 4

cd ..
mkdir -p gempic
cd gempic
cmake -D AMReX_ROOT=$AMREX_DIRECTORY/installdir -D CMAKE_C_COMPILER=mpicc -D CMAKE_CXX_COMPILER=mpicxx -D CMAKE_BUILD_TYPE=Release $SOURCE_DIRECTORY
make -j 4
