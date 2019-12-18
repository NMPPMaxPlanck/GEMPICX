# gempic

New gempic code based on AMReX

Install AMReX first enabeling particles (for each dimension):  
Irene's PC  
cd build/amrex3D  
cmake -DDIM=3 -D ENABLE_PARTICLES=ON ~/Documents/Projects/warpx_directory/amrex/


How to build the code
=====================
0.) Create a build directory  
1.) select desired amrex-dimension-version:  
cd ~build/amrex3D  
make install  
2.) Run a cmake configuration that specifies the directory where AMReX is installed and the directory where the gempic code is located  
Option with intel compiler:  
cmake -D AMReX_ROOT=/u/kako/amrex/installdir -D CMAKE_C_COMPILER=mpicc -D CMAKE_CXX_COMPILER=mpicxx -D CMAKE_CXX_FLAGS="-std=c++14" -D CMAKE_FC_COMPILER=mpif95 ../../gempic/

Option with gcc:  
cmake -D AMReX_ROOT=/u/kako/amrex/installdir -D CMAKE_C_COMPILER=mpicc -D CMAKE_CXX_COMPILER=mpicxx -D CMAKE_CXX_FLAGS="-std=c++11" -D CMAKE_FC_COMPILER=mpif95 ~/gempic/


Irene's PC:  
cmake -D AMReX_ROOT=~/Documents/Projects/warpx_directory/amrex/installdir -D CMAKE_C_COMPILER=mpicc -D CMAKE_CXX_COMPILER=mpicxx -D CMAKE_CXX_FLAGS="-std=c++11" -D CMAKE_FC_COMPILER=mpif95 ~/Documents/Projects/gempic

3.) Build the library  
make

Changing dimension
=====================
cd ~build/amrex1D  
make install  
cd ~build/gempic  
make

