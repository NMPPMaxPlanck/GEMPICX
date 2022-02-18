# Compile script for ubuntu desktop


SOURCE_DIRECTORY=`pwd`/../
SOURCE_DIRECTORY=`readlink -f $SOURCE_DIRECTORY`

echo $SOURCE_DIRECTORY

git submodule init
git submodule update

BUILD_DIR=$HOME/gempic_obj

mkdir -p $BUILD_DIR
cd $BUILD_DIR

mkdir -p gempic
cd gempic
cmake -D CMAKE_BUILD_TYPE=Release $SOURCE_DIRECTORY
make -j 2
