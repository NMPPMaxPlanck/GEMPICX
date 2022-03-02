#!/bin/bash

SOURCE_DIRECTORY=`pwd`/../
SOURCE_DIRECTORY=`readlink -f $SOURCE_DIRECTORY`
AMREX_DIRECTORY=`pwd`/../../amrex
AMREX_DIRECTORY=`readlink -f $AMREX_DIRECTORY`

echo $SOURCE_DIRECTORY
echo $AMREX_DIRECTORY

source mpcdf_modules.inc

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
