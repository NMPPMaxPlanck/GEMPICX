source /etc/profile.d/modules.sh
SOURCE_DIRECTORY=`pwd`/../
SOURCE_DIRECTORY=`readlink -f $SOURCE_DIRECTORY`
AMREX_DIRECTORY=`pwd`/../../amrex
AMREX_DIRECTORY=`readlink -f $AMREX_DIRECTORY`

echo $SOURCE_DIRECTORY
echo $AMREX_DIRECTORY

module purge
module load gcc/10
module load cmake/3.22
module load cuda/11.2
module load openmpi
module load numdiff/5.9

export AMREX_CUDA_ARCH=Ampere

BUILD_DIR=~/gempic_gpu_obj

mkdir -p $BUILD_DIR
cd $BUILD_DIR

mkdir -p amrex
cd amrex

cmake -D AMReX_GPU_BACKEND=CUDA -D AMReX_SPACEDIM=3 -D AMReX_PARTICLES=ON $AMREX_DIRECTORY

make -j 16 install

cd ..
#rm -rf gempic
mkdir -p gempic
cd gempic

cmake -D AMReX_DIR=$AMREX_DIRECTORY/installdir -D USE_CUDA=ON -D CMAKE_CUDA_ARCHITECTURES=80 -D CMAKE_BUILD_TYPE=Release $SOURCE_DIRECTORY
make -j 16
#make VERBOSE=1 -j 16
