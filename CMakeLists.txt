# Minimum version of CMake required to build this project
cmake_minimum_required(VERSION 3.14.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
include(cmake/gempic_FetchContent_Declare.cmake)
# Name of the project
project("gempic")
include(cmake/gempic_version.cmake)

option(GEMPIC_USE_CUDA "Use CUDA Backend of AMReX" OFF)
option(GEMPIC_USE_OMP "Use OpenMP Backend of AMReX (Not recommended due to reduced performance)" OFF)
option(GEMPIC_BUILD_TESTS "Build the tests. Currently only toggles unit tests." ON)
option(GEMPIC_BUILD_EXAMPLES "Build the examples. Currently only toggles all or none." ON)

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
if(   ($ENV{HOST} MATCHES "raven")
   OR ($ENV{HOST} MATCHES "cobra")
   OR (DEFINED ENV{GITLAB_CI})
   )
  set(CMAKE_BUILD_RPATH $ENV{GCC_HOME}/lib64)
  set(CMAKE_INSTALL_RPATH $ENV{GCC_HOME}/lib64)
endif()

if(GEMPIC_BUILD_TESTS)
  # Give verbose output when tests fail
  list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
  set(CMAKE_CTEST_COMMAND ${CMAKE_CTEST_COMMAND} ${CMAKE_CTEST_ARGUMENTS})
  message("ctest command is: ${CMAKE_CTEST_COMMAND}")
  # Enable ctests
  enable_testing()
  # This defines how the ctests are run
  message("HOST is $ENV{HOST}")
  if(($ENV{HOST} MATCHES "cobra*") OR ($ENV{HOST} MATCHES "raven*") OR ($ENV{HOST} MATCHES "daint*"))
    include(cmake/cobra_ctest.cmake)
  else()
    include(cmake/define_ctest.cmake)
  endif()
endif()

include(cmake/gempic_AMReXConfig.cmake)

add_library(GEMPICX STATIC)
set_target_properties(GEMPICX PROPERTIES VERSION ${GEMPIC_VERSION})

target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/Utils")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/FieldSolvers")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/FieldSolvers/Poisson")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/FieldSolvers/PoissonSolvers")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/Filter")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/Forms")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/Forms/FDDeRhamComplex")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/ParticleMeshCoupling")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/Io/Diagnostics")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/Io/Diagnostics/ReducedDiagnostics")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/Io/Diagnostics/FullDiagnostics")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/Io/Diagnostics/FlushFormats")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/Io/Diagnostics/ParticleDiag")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/Io/Diagnostics/ComputeDiagFunctors")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/Io/Checkpointing")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/Io/Profiling")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/Io/Parameters")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/Particle")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/TimeLoop")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/TimeLoop/HamiltonianSplitting")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}")

# Add subfolder with source files of the library
add_subdirectory(Src)
# Add doxygen documentation
add_subdirectory(doc)
if(GEMPIC_BUILD_TESTS)
  # Add subfolder with tests
  add_subdirectory(Testing)
endif()
if(GEMPIC_BUILD_EXAMPLES)
  include_directories("${CMAKE_SOURCE_DIR}/Examples")
  include_directories("${CMAKE_SOURCE_DIR}/Examples/QuasiNeutral")
  # Add subfolder with examples
  add_subdirectory(Examples)
endif()

if(GEMPIC_USE_CUDA)
  setup_target_for_cuda_compilation(GEMPICX)
endif()
target_link_libraries(GEMPICX PUBLIC AMReX::amrex_${AMReX_SPACEDIM}d AMReX::Flags_CXX)

# Config summary
message(STATUS "GEMPIC configuration summary: ")
message(STATUS "   AMReX version            = ${AMREX_PKG_VERSION}")
message(STATUS "   GEMPIC version           = ${GEMPIC_VERSION}")
message(STATUS "   Build type               = ${CMAKE_BUILD_TYPE}")
message(STATUS "   C++ compiler             = ${CMAKE_CXX_COMPILER}")
if(${CMAKE_BUILD_TYPE} MATCHES "Release")
  message(STATUS "   C++ flags                = ${CMAKE_CXX_FLAGS_RELEASE}")
elseif(${CMAKE_BUILD_TYPE} MATCHES "Debug")
  message(STATUS "   C++ flags                = ${CMAKE_CXX_FLAGS_DEBUG}")
endif()
if(GEMPIC_USE_CUDA)
message(STATUS "   CUDA compiler            = ${CMAKE_CUDA_COMPILER}")
message(STATUS "   CUDA flags               = ${CMAKE_CUDA_FLAGS}")
endif()

# Add gtest
gempic_FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/googletest-src
)
FetchContent_MakeAvailable(googletest)

# Preparations necessary for clang-tidy
include(cmake/clang_tidy.cmake)
