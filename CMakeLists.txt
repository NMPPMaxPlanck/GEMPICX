# Minimum version of CMake required to build this project
cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 14)

# Name of the project
project(gempic)

option(USE_CUDA "Use CUDA" OFF)

# Enable Languages
enable_language(CXX)
if (USE_CUDA)
  enable_language(CUDA)
  set(CUDA_SEPARABLE_COMPILATION ON)
  find_package(CUDA QUIET REQUIRED)
endif()

find_package(MPI REQUIRED CXX)

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

message("####### ${CMAKE_CXX_COMPILER_ID}")
IF (CMAKE_CXX_COMPILER_ID MATCHES "GNU.*")
  # gcc
  set (CMAKE_CXX_FLAGS "")
  set (CMAKE_CXX_FLAGS_RELEASE "-g -funroll-all-loops -O3 -mavx2  -march=native")
  set (CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -Wall")
  message ("Found GNU compiler " ${CMAKE_CXX_COMPILER})
ELSEIF (CMAKE_CXX_COMPILER_ID MATCHES "Intel.*")
  set (CMAKE_CXX_FLAGS "")
  set (CMAKE_CXX_FLAGS_RELEASE "-g -funroll-all-loops -O3 -xHost -zmm-usage=high -Ofast -qopt-prefetch=2 -mcmodel=large") # or try 4 for prefetch
  #set (CMAKE_CXX_FLAGS_RELEASE "-g -O3 -xHOST")
  set (CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -Wall")
  message ("Found Intel compiler " ${CMAKE_CXX_COMPILER})
ELSE()
  set (CMAKE_CXX_FLAGS "")
  set (CMAKE_CXX_FLAGS_RELEASE "")
  set (CMAKE_CXX_FLAGS_DEBUG   "")
  message("Unknown compiler")
ENDIF()

IF (USE_CUDA)
  set (CMAKE_CUDA_FLAGS "--expt-extended-lambda --expt-relaxed-constexpr -I$CUDA_HOME/include -L$CUDA_HOME/lib64 -lcurand")
  set (CMAKE_CUDA_FLAGS_RELEASE "--expt-extended-lambda --expt-relaxed-constexpr -I$CUDA_HOME/include -L$CUDA_HOME/lib64 -lcurand")
  set (CMAKE_CUDA_FLAGS_DEBUG   "--expt-extended-lambda --expt-relaxed-constexpr -I$CUDA_HOME/include -L$CUDA_HOME/lib64 -lcurand")
ENDIF()

# Enable ctests
enable_testing()
# This defines how the ctests are run
message ("HOST is $ENV{HOST}" )
IF (($ENV{HOST} MATCHES "cobra*") OR ($ENV{HOST} MATCHES "raven*"))
  include(cmake/cobra_ctest.cmake)
ELSE()
  include(cmake/define_ctest.cmake)
ENDIF()

include_directories(src)
include_directories("${CMAKE_SOURCE_DIR}/src/utils")
include_directories("${CMAKE_SOURCE_DIR}/src/field_solvers")
include_directories("${CMAKE_SOURCE_DIR}/src/field_solvers/poisson_solvers")
include_directories("${CMAKE_SOURCE_DIR}/src/forms")
include_directories("${CMAKE_SOURCE_DIR}/src/particle_mesh_coupling")
include_directories("${CMAKE_SOURCE_DIR}/src/io/diagnostics")
include_directories("${CMAKE_SOURCE_DIR}/third_party/tinyexpr")
include_directories("${CMAKE_SOURCE_DIR}/src/io/checkpointing")
include_directories("${CMAKE_SOURCE_DIR}/src/particle")
include_directories("${CMAKE_SOURCE_DIR}/src/io/profiling")
include_directories("${CMAKE_SOURCE_DIR}/src/time_loop")
include_directories("${CMAKE_SOURCE_DIR}")

add_subdirectory(third_party)
# Add subfolder with source files of the library
add_subdirectory(src)
# Add subfolder with simulations
add_subdirectory(simulations)
# Add doxygen documentation
add_subdirectory(doc)

# Config summary
message( STATUS "GEMPIC configuration summary: ")
message( STATUS "   Build type               = ${CMAKE_BUILD_TYPE}")
message( STATUS "   C++ compiler             = ${CMAKE_CXX_COMPILER}")
if ( ${CMAKE_BUILD_TYPE} MATCHES "Release" )
  message( STATUS "   C++ flags                = ${CMAKE_CXX_FLAGS_RELEASE}")
elseif( ${CMAKE_BUILD_TYPE} MATCHES "Debug" )
  message( STATUS "   C++ flags                = ${CMAKE_CXX_FLAGS_RELEASE}")
endif()
if (USE_CUDA)
message( STATUS "   CUDA compiler            = ${CMAKE_CUDA_COMPILER}")
message( STATUS "   CUDA flags               = ${CMAKE_CUDA_FLAGS}")
endif()
