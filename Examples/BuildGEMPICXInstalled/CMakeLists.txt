# Example of a project linking to an installed version of GEMPICX
# Here it's assumed that CMake can find the installation, meaning either the installation is in
# PATH or provided via CMAKE_INSTALL_PREFIX, GEMPICX_ROOT or GEMPICX_DIR

# GEMPICX minimally requires 3.24 right now, but your project can set it higher
cmake_minimum_required(VERSION 3.24)

option(USE_CUDA "Using CUDA or not" OFF)

project("FindPackageUseGempic")

# GEMPICX requires C++20
set(CMAKE_CXX_STANDARD 20)

find_package(GEMPICX REQUIRED)

if(USE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES ${AMREX_CUDA_ARCHS})
  endif()
  if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  endif()
  enable_language(CUDA)
endif()

if(   ($ENV{HOSTNAME} MATCHES "raven*")
   OR ($ENV{HOSTNAME} MATCHES "viper*")
   OR (DEFINED ENV{MPCDF_RUNNER}))
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang.*")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gcc-toolchain=$ENV{GCC_HOME}" CACHE STRING "Compile flags" FORCE)
  endif()
endif()

add_executable(example UseInstalledGEMPICX.cpp)

if(USE_CUDA)
  # Hard coded because we have 1 project with 1 file. Not necessary at all if we use KOKKOS.
  set_target_properties(example PROPERTIES CUDA_SEPARABLE_COMPILATION ON) # This adds -dc
  set_source_files_properties(UseInstalledGEMPICX.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(example PROPERTIES CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
endif()
target_link_libraries(example GEMPICX::GEMPICX)

enable_testing()
add_test(NAME GEMPICXInstalled_Verify COMMAND example)
