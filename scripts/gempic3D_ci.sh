#!/bin/bash

#set -x
set -e

# ---
# Build gempic on Ubuntu
# ---

# # build in folder parallel to gempic source (not src) folder
# BUILD_DIR=${BUILD_DIR:=$(realpath $(dirname $BASH_SOURCE)/../../gempic_obj)}
# build in /tmp
BUILD_DIR=${BUILD_DIR:=/tmp/gempic_obj}
# processors used for parallel build
MAKE_NPROCS="${MAKE_NPROCS:-4}"
# define the CMAKE build type
CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:="Release"}

GEMPIC_BASE=`readlink -f ..`

git submodule sync
git submodule init
git submodule update

mkdir -p $BUILD_DIR
cd $BUILD_DIR

cmake $GEMPIC_BASE \
      -D AMReX_DIR=$BUILD_DIR/amrex/installdir \
      -D CMAKE_C_COMPILER=mpicc \
      -D CMAKE_CXX_COMPILER=mpicxx \
      -D CMAKE_CXX_FLAGS="-std=c++17" \
      -D CMAKE_FC_COMPILER=mpif95

make -j $MAKE_NPROCS

ctest --verbose
