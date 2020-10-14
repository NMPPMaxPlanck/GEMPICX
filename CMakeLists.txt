# Minimum version of CMake required to build this project
cmake_minimum_required(VERSION 3.0)

# Name of the project
project(gempic)

find_package(MPI REQUIRED)

message("AMReX Dir is ${AMReX_ROOT}")

# Find the AMReX library
find_package( AMReX REQUIRED HINTS ${AMReX_ROOT}/lib/cmake/AMReX )
# This is an old version to link with amrex (gfortran and intel versions, respectively
#set( AMReX_LIBRARIES ${AMReX_ROOT}/lib/libamrex.a;gfortran )
#set( AMReX_LIBRARIES ${AMReX_ROOT}/lib/libamrex.a;ifcore )
include_directories( ${AMReX_ROOT}/include )
#message("AMReX_LIBRARIES ${AMReX_LIBRARIES}")
#message("AMReX Dir is ${AMReX_DIR} ${AMReX_ROOT}")
#set( AMREX ON CACHE BOOL "AMREX found." )

# Set compiler flags

message("####### ${CXX_COMPILER_NAME}")
IF (CXX_COMPILER_NAME MATCHES "gcc.*")
  # gcc
  set (CMAKE_CXX_FLAGS "")
  set (CMAKE_CXX_FLAGS_RELEASE "-g -funroll-all-loops -O3 -mavx2  -march=native")
  set (CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -Wall")
  message ("Found gcc compiler " ${CMAKE_CXX_COMPILER})
ELSEIF (CXX_COMPILE_NAME MATCHES "icpc")
  set (CMAKE_CXX_FLAGS "")
  set (CMAKE_CXX_FLAGS_RELEASE "-g -funroll-all-loops -O3 -xCORE-AVX512 -zmm-usage=high -Ofast -qopt-prefetch=2 -mcmodel=large") # or try 4 for prefetch
  #set (CMAKE_CXX_FLAGS_RELEASE "-g -O3 -xHOST")
  set (CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -Wall")
  message ("Found icpc compiler " ${CMAKE_CXX_COMPILER})
ENDIF()

# Enable ctests
enable_testing()
# This defines how the ctests are run
include(cmake/define_ctest.cmake)

# Add subfolder with source files of the library
add_subdirectory(src)
# Add subfolder with simulations
add_subdirectory(simulations)
# Add doxygen documentation
add_subdirectory(doc)

