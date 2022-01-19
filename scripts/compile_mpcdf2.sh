#!/usr/bin/sh
SOURCE_DIRECTORY=`pwd`/../
SOURCE_DIRECTORY=`readlink -f $SOURCE_DIRECTORY`
AMREX_DIRECTORY=`pwd`/../../amrex
AMREX_DIRECTORY=`readlink -f $AMREX_DIRECTORY`

echo $SOURCE_DIRECTORY
echo $AMREX_DIRECTORY

#cp $SOURCE_DIRECTORY/third_party/AMReX_MLNodeLap_Gempic_3D_K.H $AMREX_DIRECTORY/Src/LinearSolvers/MLMG/AMReX_MLNodeLap_3D_K.H

source /etc/profile.d/modules.sh
module purge
if [ x"$CLUSTER" == x"COBRA"  ]; then
module load intel/19.1.1
module load impi/2019.7
module load mkl/2019.5
fi
if [ x"$CLUSTER" == x"RAVEN" ]; then
module load intel/19.1.2
module load impi/2019.8
module load mkl/2021.2
fi
module load cmake/3.18

BUILD_DIR=$HOME/gempic_obj

mkdir -p $BUILD_DIR
cd $BUILD_DIR

mkdir -p amrex_install
mkdir -p amrex
cd amrex

cmake -D AMReX_SPACEDIM=3 -D AMReX_PARTICLES=ON -D CMAKE_BUILD_TYPE=Release -D CMAKE_C_COMPILER=mpiicc -D CMAKE_CXX_COMPILER=mpiicpc  -D CMAKE_LINKER=mpiicpc -D CMAKE_INSTALL_PREFIX=$BUILD_DIR/amrex_install $AMREX_DIRECTORY

make -j 10 install

cd ..
mkdir -p gempic
cd gempic
cmake  -D AMReX_DIR=$BUILD_DIR/amrex_install -D CMAKE_C_COMPILER=mpiicc -D CMAKE_CXX_COMPILER=mpiicpc  -D CMAKE_BUILD_TYPE=Release  $SOURCE_DIRECTORY
make VERBOSE=1 -j 10
