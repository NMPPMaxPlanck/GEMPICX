#!/bin/bash

# ---
# Build gempic on Ubuntu
# ---

# build in /tmp
BUILD_DIR=${BUILD_DIR:=/tmp/gempic_obj}
# define the CMAKE build type
CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:="Release"}

GEMPIC_BASE=`readlink -f ..`

mkdir -p $BUILD_DIR
cd $BUILD_DIR

# now download amrex
git clone https://github.com/AMReX-Codes/amrex.git
# install amrex
mkdir build_amrex
cd build_amrex
cmake -DDIM=3 -D ENABLE_PARTICLES=ON $BUILD_DIR/amrex
make install
cd ..

#export CC=gcc
#export CXX=g++

cmake $GEMPIC_BASE \
      -D AMReX_ROOT=$BUILD_DIR/amrex/installdir \
      -D CMAKE_C_COMPILER=mpicc \
      -D CMAKE_CXX_COMPILER=mpicxx \
      -D CMAKE_CXX_FLAGS="-std=c++11" \
      -D CMAKE_FC_COMPILER=mpif95

make -j 4

