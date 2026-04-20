# Example of a project linking to a fetched version of GEMPICX
# Here it's assumed that CMake can find the repository, meaning either the repository is in
# some given gempicx_SOURCE_DIR or to be downloaded via the online git repository

# GEMPICX minimally requires 3.24 right now, but your project can set it higher
cmake_minimum_required(VERSION 3.24)

option(USE_CUDA "Using CUDA or not" OFF)

project("FetchContentUseGempic")
# GEMPICX requires C++20
set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
# Would obviously look different in your project, e.g.
# FetchContent_Declare(GEMPICX
#     GIT_REPOSITORY https://github.com/NMPPMaxPlanck/GEMPICX.git
#     GIT_TAG 0.5.552
# )
FetchContent_Declare(GEMPICX
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/../..
)
# Set GEMPICX options here ...
FetchContent_MakeAvailable(GEMPICX)

if(USE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES ${AMREX_CUDA_ARCHS})
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

add_executable(example UseFetchedGEMPICX.cpp)

if(USE_CUDA)
  # Hard coded because we have 1 project with 1 file. Not necessary at all if we use KOKKOS.
  set_target_properties(example PROPERTIES CUDA_SEPARABLE_COMPILATION ON) # This adds -dc
  set_source_files_properties(UseFetchedGEMPICX.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(example PROPERTIES CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
endif()
target_link_libraries(example GEMPICX::GEMPICX)

enable_testing()
add_test(NAME GEMPICXFetched_Verify COMMAND example)
