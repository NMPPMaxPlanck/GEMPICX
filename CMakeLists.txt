# Minimum version of CMake required to build this project
cmake_minimum_required(VERSION 3.14.0)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# Name of the project
project("gempic")

option(GEMPIC_USE_CUDA "Use CUDA" OFF)
option(GEMPIC_USE_OMP "Use OpenMP" OFF)

# Enable Languages
enable_language(CXX)
if(GEMPIC_USE_CUDA)
  enable_language(CUDA)
  set(CUDA_SEPARABLE_COMPILATION ON)
  find_package(CUDA QUIET REQUIRED)
endif()

# Set compiler flags
include(cmake/gempic_CompilerConfig.cmake)


# Give verbose output when tests fail
list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
message("ctest should be: ${CMAKE_CTEST_COMMAND}")
set(CMAKE_CTEST_COMMAND ${CMAKE_CTEST_COMMAND} ${CMAKE_CTEST_ARGUMENTS})
message("ctest should be: ${CMAKE_CTEST_COMMAND}")
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

include_directories("${CMAKE_SOURCE_DIR}/src")
include_directories("${CMAKE_SOURCE_DIR}/src/utils")
include_directories("${CMAKE_SOURCE_DIR}/src/field_solvers")
include_directories("${CMAKE_SOURCE_DIR}/src/field_solvers/Poisson")
include_directories("${CMAKE_SOURCE_DIR}/src/field_solvers/poisson_solvers")
include_directories("${CMAKE_SOURCE_DIR}/src/forms")
include_directories("${CMAKE_SOURCE_DIR}/src/forms/FDDeRhamComplex")
include_directories("${CMAKE_SOURCE_DIR}/src/particle_mesh_coupling")
include_directories("${CMAKE_SOURCE_DIR}/src/io/diagnostics")
include_directories("${CMAKE_SOURCE_DIR}/src/io/diagnostics/ReducedDiags")
include_directories("${CMAKE_SOURCE_DIR}/src/io/checkpointing")
include_directories("${CMAKE_SOURCE_DIR}/src/particle")
include_directories("${CMAKE_SOURCE_DIR}/src/io/profiling")
include_directories("${CMAKE_SOURCE_DIR}/src/io/parameters")
include_directories("${CMAKE_SOURCE_DIR}/src/simulations")
include_directories("${CMAKE_SOURCE_DIR}/src/time_loop")
include_directories("${CMAKE_SOURCE_DIR}/src/time_loop/HamiltonianSplitting")
include_directories("${CMAKE_SOURCE_DIR}")

# Add subfolder with source files of the library
add_subdirectory(src)
# Add subfolder with simulations
add_subdirectory(simulations)
# Add doxygen documentation
add_subdirectory(doc)
# Add subfolder with tests
add_subdirectory(Testing)

# Config summary
message( STATUS "GEMPIC configuration summary: ")
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
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/b796f7d44681514f58a683a3a71ff17c94edb0c1.zip
)
FetchContent_MakeAvailable(googletest)