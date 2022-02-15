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
module load intel/21.4.0
module load impi/2021.4
module load mkl/2021.2
fi
module load cmake/3.18
module load gcc

echo "#-------------------------------------------#"
echo "#---------- Compilers ----------------------#"
echo "#-------------------------------------------#"
echo
echo "#-------------------------------------------#"
echo " -- Base CPU compiler C/C++"
ICC=$(which icc)
IXX=$(which icpc)
export CC=${CC:-$ICC}
export CXX=${CXX:-$IXX}
export MPICXX=$(which mpiicpc)
echo $CC
echo $CXX
echo $MPICXX
echo

BUILD_DIR=$HOME/gempic_obj

mkdir -p $BUILD_DIR
cd $BUILD_DIR

mkdir -p amrex
cd amrex

cmake -D AMReX_SPACEDIM=3 -D AMReX_PARTICLES=ON -D CMAKE_BUILD_TYPE=Release $AMREX_DIRECTORY

make -j 10 install

cd ..
mkdir -p gempic
cd gempic
cmake  -D AMReX_DIR=$AMREX_DIRECTORY/installdir -D CMAKE_BUILD_TYPE=Release $SOURCE_DIRECTORY
make VERBOSE=1 -j 10
