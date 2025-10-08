cmake_minimum_required(VERSION 3.24.0)

file(REAL_PATH ${CMAKE_BINARY_DIR} REAL_CMAKE_BINARY_DIR)
file(REAL_PATH ${CMAKE_SOURCE_DIR} REAL_CMAKE_SOURCE_DIR)
if(${REAL_CMAKE_BINARY_DIR} STREQUAL ${REAL_CMAKE_SOURCE_DIR})
  message(FATAL_ERROR "GEMPIC: Building in source tree is not allowed.\n"
                      "        Delete CMakeCache.txt and CMakeFiles.\n"
                      "        Use `cmake -B /PATH/TO/BUILD/DIR ...` to set a different binary directory.")
endif()

option(GEMPIC_BUILD_ONLY_DOCUMENTATION "Build only documentation." OFF)

if(GEMPIC_BUILD_ONLY_DOCUMENTATION)
  project("gempic-doc" LANGUAGES)
  include(cmake/gempic_version.cmake)
  add_subdirectory(doc)
else()

project("gempic")
include(cmake/gempic_version.cmake)

option(GEMPIC_USE_CUDA "Use CUDA Backend of AMReX" OFF)
option(GEMPIC_USE_HIP "Use HIP Backend of AMReX" OFF)
option(GEMPIC_USE_OMP "Use OpenMP Backend of AMReX (Not recommended due to reduced performance)" OFF)
if(GEMPIC_USE_CUDA OR GEMPIC_USE_HIP OR AMReX_SPACEDIM EQUAL 1)
  option(GEMPIC_USE_HYPRE "Use Hypre solver for AMReX" OFF)
  if(GEMPIC_USE_HYPRE)
    message(FATAL_ERROR "The current HYPRE interface of GEMPIC is neither GPU nor 1D compatible. Set -D GEMPIC_USE_HYPRE=OFF in CMake.")
  endif()
else()
  option(GEMPIC_USE_HYPRE "Use Hypre solver for AMReX" ON)
endif()

option(GEMPIC_USE_LTO "Use Link-Time Optimization. Turning it off reduces compile time, but may reduce performance" OFF)
option(GEMPIC_BUILD_TESTS "Build the tests. Currently only toggles unit tests." ON)
option(GEMPIC_BUILD_EXAMPLES "Build the examples. Currently only toggles all or none." ON)
option(GEMPIC_BUILD_DOCUMENTATION "Add the documentation to the build." OFF)

option(GEMPIC_PROFILE_KERNELS "Enable tiny profiling in host device functions when executed on host." OFF)

include(FetchContent)
include(cmake/gempic_FetchContent_Declare.cmake)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Enable Languages
enable_language(CXX)
if(GEMPIC_USE_CUDA)
  enable_language(CUDA)
  set(CUDA_SEPARABLE_COMPILATION ON)
endif()
if(GEMPIC_USE_HIP)
  enable_language(HIP)
  # note. this line here because it is there for CUDA.
  # Option is irrelevant as of 2025-05-20
  # (see https://gitlab.kitware.com/cmake/cmake/-/issues/23210)
  # set(HIP_SEPARABLE_COMPILATION ON)
endif()

# Set compiler flags
include(cmake/gempic_CompilerConfig.cmake)

# Define cmake to use libraries from gcc-module not default gcc
# This section is specific for MPCDF-Cluster configurations
if(   ($ENV{HOST} MATCHES "raven*")
   OR ($ENV{HOST} MATCHES "viper*")
   OR (DEFINED ENV{MPCDF_RUNNER})
   )
  set(CMAKE_BUILD_RPATH $ENV{GCC_HOME}/lib64)
  set(CMAKE_INSTALL_RPATH $ENV{GCC_HOME}/lib64)
endif()

include(cmake/gempic_AMReXConfig.cmake)

add_library(GEMPICX STATIC)
set_target_properties(GEMPICX PROPERTIES VERSION ${GEMPIC_VERSION})
if(GEMPIC_PROFILE_KERNELS)
  if(not AMReX_TINY_PROFILE)
    message(WARNING "GEMPIC_PROFILE_KERNELS does nothing if AMReX_TINY_PROFILE is not turned on")
  endif()
  target_compile_definitions(GEMPICX PRIVATE GEMPIC_PROFILE_KERNELS)
endif()

target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/DataStructures")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/FieldSolvers")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/FieldSolvers/Poisson")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/FieldSolvers/Quasineutral")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/Filter")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/Forms")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/Forms/FDDeRhamComplex")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/HyperbolicSolvers")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/Io/Diagnostics")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/Io/Diagnostics/FullDiagnostics")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/Io/Diagnostics/FullDiagnostics/DiagnosticOutputProcessors")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/Io/Diagnostics/FullDiagnostics/FlushFormats")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/Io/Diagnostics/ReducedDiagnostics")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/Io/Parameters")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/Io/Profiling")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/Particle")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/ParticleMeshCoupling")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/TimeLoop")
target_include_directories(GEMPICX PUBLIC "${CMAKE_SOURCE_DIR}/Src/Utils")

# Add subfolder with source files of the library
add_subdirectory(Src)
if(GEMPIC_BUILD_DOCUMENTATION)
  add_subdirectory(doc)
endif()
if(GEMPIC_BUILD_EXAMPLES)
  # Add subfolder with examples
  add_subdirectory(Examples)
endif()
if(GEMPIC_BUILD_TESTS)
  # Enable ctest and GoogleTest tests
  enable_testing()
  # Add subfolder with tests
  add_subdirectory(Testing)
endif()

if(GEMPIC_USE_CUDA)
  # function defined in AMReX tools. No equivalent for HIP
  setup_target_for_cuda_compilation(GEMPICX)
endif()
target_link_libraries(GEMPICX PUBLIC AMReX::amrex_${AMReX_SPACEDIM}d AMReX::Flags_CXX)

# Add cmake-git-version-tracking
gempic_FetchContent_Declare(
  cmake_git_version_tracking
  GIT_REPOSITORY https://github.com/andrew-hardin/cmake-git-version-tracking.git
  GIT_TAG 6c0cb87edd029ddfb403a8e24577c144a03605a6
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/cmake_git_version_tracking-src
  )
  FetchContent_MakeAvailable(cmake_git_version_tracking)
  
# Preparations necessary for clang-tidy
include(cmake/clang_tidy.cmake)

# Config summary
message(STATUS "GEMPIC configuration summary: ")
message(STATUS "   AMReX version            = ${AMReX_VERSION_USED}")
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
if(GEMPIC_USE_HIP)
  message(STATUS "   HIP compiler            = ${CMAKE_HIP_COMPILER}")
  message(STATUS "   HIP flags               = ${CMAKE_HIP_FLAGS}")
endif()

endif() #CMAKE_BUILD_ONLY_DOCUMENTATION OFF
