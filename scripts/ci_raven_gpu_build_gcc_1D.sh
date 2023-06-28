#!/bin/bash

set -e

SOURCE_DIRECTORY=`pwd`/../
SOURCE_DIRECTORY=`readlink -f $SOURCE_DIRECTORY`
AMREX_DIRECTORY=`pwd`/../third_party/amrex
AMREX_DIRECTORY=`readlink -f $AMREX_DIRECTORY`


echo $SOURCE_DIRECTORY


source ci_raven_gpu_build_gcc.inc
export CC=`which gcc`
export CXX=`which g++`

MAKE_NPROCS="${MAKE_NPROCS:-16}"

git submodule sync
git submodule init
git submodule update

BUILD_DIR="${BUILD_DIR:-../build}"

# WARNING: We purge the build dir completely, in case this is not wanted,
# comment the line below and enable `make clean` below to get full builds.
rm -rf "$BUILD_DIR"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

mkdir -p amrex
pushd amrex

export AMREX_CUDA_ARCH=Ampere
cmake -D AMReX_GPU_BACKEND=CUDA -D AMReX_SPACEDIM=1 -D AMReX_PARTICLES=ON $AMREX_DIRECTORY
#make clean
make -j $MAKE_NPROCS #VERBOSE=1
make install

popd


mkdir -p gempic
pushd gempic

cmake -D AMReX_DIR=$AMREX_DIRECTORY/installdir -D GEMPIC_USE_CUDA=ON -D CMAKE_CUDA_ARCHITECTURES=80 -D CMAKE_BUILD_TYPE=Release $SOURCE_DIRECTORY
#make clean
make -j $MAKE_NPROCS #VERBOSE=1

popd
