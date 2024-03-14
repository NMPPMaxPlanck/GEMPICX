# Minimum version of CMake required to build this project
cmake_minimum_required(VERSION 3.14.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# Name of the project
project("gempic")

option(GEMPIC_USE_CUDA "Use CUDA Backend of AMReX" OFF)
option(GEMPIC_USE_OMP "Use OpenMP Backend of AMReX (Not recommended due to reduced performance)" OFF)

# Enable Languages
enable_language(CXX)
if(GEMPIC_USE_CUDA)
  enable_language(CUDA)
  set(CUDA_SEPARABLE_COMPILATION ON)
  find_package(CUDA QUIET REQUIRED)
endif()

# Set compiler flags
include(cmake/gempic_CompilerConfig.cmake)

# Define cmake to use libraries from gcc-module not default gcc
# This section is specific for MPCDF-Cluster configurations
IF(   ($ENV{HOST} MATCHES "raven")
   OR ($ENV{HOST} MATCHES "cobra")
   OR (DEFINED ENV{GITLAB_CI})
   )
  SET( CMAKE_BUILD_RPATH $ENV{GCC_HOME}/lib64 )
  SET( CMAKE_INSTALL_RPATH $ENV{GCC_HOME}/lib64 )
ENDIF()

# Give verbose output when tests fail
list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
set(CMAKE_CTEST_COMMAND ${CMAKE_CTEST_COMMAND} ${CMAKE_CTEST_ARGUMENTS})
message("ctest command is: ${CMAKE_CTEST_COMMAND}")
# Enable ctests
enable_testing()
# This defines how the ctests are run
message("HOST is $ENV{HOST}" )
if(($ENV{HOST} MATCHES "cobra*") OR ($ENV{HOST} MATCHES "raven*") OR ($ENV{HOST} MATCHES "daint*") )
  include(cmake/cobra_ctest.cmake)
else()
  include(cmake/define_ctest.cmake)
endif()

include(cmake/gempic_AMReXConfig.cmake)

include_directories("${CMAKE_SOURCE_DIR}/Src")
include_directories("${CMAKE_SOURCE_DIR}/Src/Utils")
include_directories("${CMAKE_SOURCE_DIR}/Src/FieldSolvers")
include_directories("${CMAKE_SOURCE_DIR}/Src/FieldSolvers/Poisson")
include_directories("${CMAKE_SOURCE_DIR}/Src/FieldSolvers/PoissonSolvers")
include_directories("${CMAKE_SOURCE_DIR}/Src/Filter")
include_directories("${CMAKE_SOURCE_DIR}/Src/Forms")
include_directories("${CMAKE_SOURCE_DIR}/Src/Forms/FDDeRhamComplex")
include_directories("${CMAKE_SOURCE_DIR}/Src/ParticleMeshCoupling")
include_directories("${CMAKE_SOURCE_DIR}/Src/Io/Diagnostics")
include_directories("${CMAKE_SOURCE_DIR}/Src/Io/Diagnostics/ReducedDiagnostics")
include_directories("${CMAKE_SOURCE_DIR}/Src/Io/Diagnostics/FullDiagnostics")
include_directories("${CMAKE_SOURCE_DIR}/Src/Io/Diagnostics/FlushFormats")
include_directories("${CMAKE_SOURCE_DIR}/Src/Io/Diagnostics/ParticleDiag")
include_directories("${CMAKE_SOURCE_DIR}/Src/Io/Diagnostics/ComputeDiagFunctors")
include_directories("${CMAKE_SOURCE_DIR}/Src/Io/Checkpointing")
include_directories("${CMAKE_SOURCE_DIR}/Src/Io/Profiling")
include_directories("${CMAKE_SOURCE_DIR}/Src/Io/Parameters")
include_directories("${CMAKE_SOURCE_DIR}/Src/Particle")
include_directories("${CMAKE_SOURCE_DIR}/Src/Simulations")
include_directories("${CMAKE_SOURCE_DIR}/Src/TimeLoop")
include_directories("${CMAKE_SOURCE_DIR}/Src/TimeLoop/HamiltonianSplitting")
include_directories("${CMAKE_SOURCE_DIR}")

# Add subfolder with source files of the library
add_subdirectory(Src)
# Add doxygen documentation
add_subdirectory(doc)
# Add subfolder with tests
add_subdirectory(Testing)

# Config summary
message( STATUS "GEMPIC configuration summary: ")
message( STATUS "   AMReX version            = ${AMREX_PKG_VERSION}")
message( STATUS "   Build type               = ${CMAKE_BUILD_TYPE}")
message( STATUS "   C++ compiler             = ${CMAKE_CXX_COMPILER}")
if( ${CMAKE_BUILD_TYPE} MATCHES "Release" )
  message( STATUS "   C++ flags                = ${CMAKE_CXX_FLAGS_RELEASE}")
elseif( ${CMAKE_BUILD_TYPE} MATCHES "Debug" )
  message( STATUS "   C++ flags                = ${CMAKE_CXX_FLAGS_DEBUG}")
endif()
if(GEMPIC_USE_CUDA)
message( STATUS "   CUDA compiler            = ${CMAKE_CUDA_COMPILER}")
message( STATUS "   CUDA flags               = ${CMAKE_CUDA_FLAGS}")
endif()

# Add gtest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
  SOURCE_DIR "${CMAKE_SOURCE_DIR}/third_party/googletest-src"
)
FetchContent_MakeAvailable(googletest)

# Preparations necessary for clang-tidy
include(cmake/clang_tidy.cmake)
