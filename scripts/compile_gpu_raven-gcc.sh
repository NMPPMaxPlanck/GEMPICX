source /etc/profile.d/modules.sh
SOURCE_DIRECTORY=`pwd`/../
SOURCE_DIRECTORY=`readlink -f $SOURCE_DIRECTORY`
AMREX_DIRECTORY=`pwd`/../third_party/amrex
AMREX_DIRECTORY=`readlink -f $AMREX_DIRECTORY`

echo $SOURCE_DIRECTORY
echo $AMREX_DIRECTORY

module purge
module load gcc/11
module load cmake/3.22
module load cuda/11.4
module load openmpi_gpu/4
module load numdiff/5.9

echo " -- Base GCC compiler C/C++"
GCC=$(which gcc)
GXX=$(which g++)
export CC=${CC:-$GCC}
export CXX=${CXX:-$GXX}
#export MPICXX=$(which mpicc)
echo $CC
echo $CXX
#echo $MPICXX
echo

MAKE_NPROCS="${MAKE_NPROCS:-16}"

git submodule init
git submodule update

export GPUS_PER_SOCKET=2
export GPUS_PER_NODE=4
export AMREX_CUDA_ARCH=Ampere

BUILD_DIR=~/gempic_gpu_obj

# comment this in if you want to switch modules or compilers, so CMake
# configures the build again from scratch instead of reusing the previous one
#rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR

mkdir -p amrex
cd amrex

cmake -D AMReX_GPU_BACKEND=CUDA -D AMReX_SPACEDIM=3 -D AMReX_PARTICLES=ON $AMREX_DIRECTORY

make -j $MAKE_NPROCS install

cd ..
mkdir -p gempic
cd gempic

cmake -D AMReX_DIR=$AMREX_DIRECTORY/installdir -D USE_CUDA=ON -D CMAKE_CUDA_ARCHITECTURES=80 -D CMAKE_BUILD_TYPE=Release $SOURCE_DIRECTORY
make -j $MAKE_NPROCS 
#make VERBOSE=1 -j $MAKE_NPROCS
