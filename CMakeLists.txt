# Minimum version of CMake required to build this project
cmake_minimum_required(VERSION 3.0)

# Name of the project
project(gempic)

find_package(MPI REQUIRED)

message("AMReX Dir is ${AMReX_ROOT}")

# Find the AMReX library
find_package( AMReX REQUIRED HINTS ${AMReX_ROOT}/lib/cmake/AMReX )
set( AMReX_LIBRARIES ${AMReX_ROOT}/lib/libamrex.a;gfortran )
include_directories( ${AMReX_ROOT}/include )
#message("AMReX_LIBRARIES ${AMReX_LIBRARIES}")
#message("AMReX Dir is ${AMReX_DIR} ${AMReX_ROOT}")
#set( AMREX ON CACHE BOOL "AMREX found." )

# Add subfolder with source files of the library
add_subdirectory(src)

