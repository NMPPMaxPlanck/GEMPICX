# gempic

New gempic code based on AMReX

Install AMReX first enabeling particles:
Irene's PC
cmake -D ENABLE_PARTICLES=ON ~/Documents/Projects/warpx_directory/amrex/


How to build the code
=====================
0.) Create a build directory
1.) Run a cmake configuration that specifies the directory where AMReX is installed and the directory where the gempic code is located
Option with intel compiler:
cmake -D AMReX_ROOT=/u/kako/amrex/installdir -D CMAKE_C_COMPILER=mpicc -D CMAKE_CXX_COMPILER=mpicxx -D CMAKE_CXX_FLAGS="-std=c++14" -D CMAKE_FC_COMPILER=mpif95 ../../gempic/

Option with gcc:
cmake -D AMReX_ROOT=/u/kako/amrex/installdir -D CMAKE_C_COMPILER=mpicc -D CMAKE_CXX_COMPILER=mpicxx -D CMAKE_CXX_FLAGS="-std=c++11" -D CMAKE_FC_COMPILER=mpif95 ~/gempic/


Irene's PC:
cmake -D AMReX_ROOT=~/Documents/Projects/warpx_directory/amrex/installdir -D CMAKE_C_COMPILER=mpicc -D CMAKE_CXX_COMPILER=mpicxx -D CMAKE_CXX_FLAGS="-std=c++11" -D CMAKE_FC_COMPILER=mpif95 ~/Documents/Projects/gempic

2.) Build the library
make
