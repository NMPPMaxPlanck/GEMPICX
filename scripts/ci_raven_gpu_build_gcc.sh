#!/bin/bash

set -e

SOURCE_DIRECTORY=`pwd`/../
SOURCE_DIRECTORY=`readlink -f $SOURCE_DIRECTORY`

echo $SOURCE_DIRECTORY


source ci_raven_gpu_build_gcc.inc
export CC=`which gcc`
export CXX=`which g++`

git submodule init
git submodule update

BUILD_DIR="${BUILD_DIR:-../build}"

# WARNING: We purge the build dir completely, in case this is not wanted,
# comment the line below and enable `make clean` below to get full builds.
rm -rf "$BUILD_DIR"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"


export AMREX_CUDA_ARCH=Ampere


mkdir -p gempic
pushd gempic

cmake -D USE_CUDA=ON -D CMAKE_CUDA_ARCHITECTURES=80 -D CMAKE_BUILD_TYPE=Release $SOURCE_DIRECTORY
#make clean
make -j 16 #VERBOSE=1

popd
