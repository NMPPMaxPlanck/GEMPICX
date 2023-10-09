#!/bin/bash

#set -x
set -e

# ---
# Build gempic on Ubuntu
# ---

# build in /tmp
BUILD_DIR=`readlink -f ../build2D`
# processors used for parallel build
MAKE_NPROCS="${MAKE_NPROCS:-4}"
# define the CMAKE build type
CMAKE_BUILD_TYPE="Debug"

GEMPIC_BASE=`readlink -f ..`

git submodule sync
git submodule init
git submodule update

mkdir -p $BUILD_DIR
cd $BUILD_DIR

# now download amrex
#git clone https://github.com/AMReX-Codes/amrex.git

# ---- 3D ----
#cp $GEMPIC_BASE/src/field_solvers/testing/test_maxwell_yee_3D.output $GEMPIC_BASE/src/field_solvers/testing/test_maxwell_yee.expected_output


# install amrex
#rm -rf build_amrex
#mkdir build_amrex
#cd build_amrex
#cmake -D AMReX_SPACEDIM=3 -D AMReX_PARTICLES=ON -D AMReX_TINY_PROFILE=ON $BUILD_DIR/amrex
#make install
#cd ..

#export CC=gcc
#export CXX=g++

cmake $GEMPIC_BASE \
      -D AMReX_DIR=$BUILD_DIR/amrex/installdir \
      -D CMAKE_C_COMPILER=mpicc \
      -D CMAKE_CXX_COMPILER=mpicxx \
      -D CMAKE_CXX_FLAGS="-std=c++17" \
      -D CMAKE_FC_COMPILER=mpif95 \
      -D CMAKE_BUILD_TYPE=Debug \
      -D AMReX_SPACEDIM=2

make -j $MAKE_NPROCS

ctest --verbose
