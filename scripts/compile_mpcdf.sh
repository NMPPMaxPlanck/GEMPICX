#!/bin/bash

SOURCE_DIRECTORY=`dirname $0`/../
SOURCE_DIRECTORY=`readlink -f $SOURCE_DIRECTORY`
AMREX_DIRECTORY=$SOURCE_DIRECTORY/third_party/amrex

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

cmake -D AMReX_GPU_BACKEND=CUDA -D AMReX_SPACEDIM=3 -D AMReX_PARTICLES=ON -D AMReX_TINY_PROFILE=ON $AMREX_DIRECTORY

make -j $MAKE_NRPOCS install

cd ..
mkdir -p gempic
cd gempic
cmake  -D AMReX_DIR=$AMREX_DIRECTORY/installdir -D CMAKE_BUILD_TYPE=Release $SOURCE_DIRECTORY
make VERBOSE=1 -j 10

# generate run script in BUILD_DIR
rm -f $BUILD_DIR/run_mpcdf.sh
cat $SOURCE_DIRECTORY/scripts/slurm_mpcdf.inc $SOURCE_DIRECTORY/scripts/mpcdf_modules.inc $SOURCE_DIRECTORY/scripts/srun.inc > $BUILD_DIR/run_mpcdf.sh
