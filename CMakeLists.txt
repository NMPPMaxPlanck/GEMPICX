cmake_minimum_required(VERSION 3.24.0)

include(cmake/gempic_utils.cmake)
include(cmake/gempic_version.cmake)

gempic_enforce_separate_build_dir()
gempic_get_version()

option(GEMPIC_BUILD_ONLY_DOCUMENTATION "Build only documentation" OFF)

if(GEMPIC_BUILD_ONLY_DOCUMENTATION)
  project("gempic-doc" LANGUAGES)
  add_subdirectory(doc)
else()
# Configure GEMPIC source project
if(GEMPICX_PKG_VERSION)
  project("gempic" VERSION ${GEMPICX_PKG_VERSION})
else()
  project("gempic")
endif()

option(GEMPIC_USE_CUDA "Use CUDA Backend of AMReX" OFF)
option(GEMPIC_USE_HIP "Use HIP Backend of AMReX" OFF)
option(GEMPIC_USE_OMP "Use OpenMP Backend of AMReX (Not recommended due to reduced performance)" OFF)
if(GEMPIC_USE_CUDA OR GEMPIC_USE_HIP OR AMReX_SPACEDIM EQUAL 1)
  if(AMReX_HYPRE)
    message(FATAL_ERROR "The current HYPRE interface of GEMPIC is neither GPU nor 1D compatible. Set -DAMReX_HYPRE=OFF in CMake.")
  endif()
else()
  set(AMReX_HYPRE ON CACHE BOOL "AMReX Option set by default within GEMPIC")
endif()

option(GEMPIC_USE_LTO "Use Link-Time Optimization. Turning it off reduces compile time, but may reduce performance." OFF)
option(GEMPIC_BUILD_TESTS "Build the tests. Currently only toggles unit tests." ON)
option(GEMPIC_BUILD_EXAMPLES "Build the examples. Currently only toggles all or none." ON)
option(GEMPIC_BUILD_DOCUMENTATION "Add the documentation to the build" OFF)

option(GEMPIC_PROFILE_KERNELS "Enable tiny profiling in host device functions when executed on host" OFF)
option(USE_DIRTY_AMREX_REPO "Allow building from dirty AMReX git repositories" OFF)
option(GEMPIC_DEBUG_MESSAGES "Turns on GEMPIC-specific debug messages" OFF)

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

include(cmake/gempic_install.cmake)
gempic_set_default_install_directories()

include(cmake/gempic_FetchContent_Declare.cmake)

# Set compiler flags
include(cmake/gempic_CompilerConfig.cmake)

include(cmake/gempic_AMReXConfig.cmake)

add_library(GEMPICX STATIC)
add_library(GEMPICX::GEMPICX ALIAS GEMPICX)
if(GEMPICX_PKG_VERSION)
  set_target_properties(GEMPICX PROPERTIES VERSION ${GEMPICX_PKG_VERSION})
endif()
if(GEMPIC_PROFILE_KERNELS)
  if(not AMReX_TINY_PROFILE)
    message(WARNING "GEMPIC_PROFILE_KERNELS does nothing if AMReX_TINY_PROFILE is not turned on")
  endif()
  target_compile_definitions(GEMPICX PRIVATE GEMPIC_PROFILE_KERNELS)
endif()
if(GEMPIC_DEBUG_MESSAGES)
  target_compile_definitions(GEMPICX PRIVATE GEMPIC_DEBUG_MESSAGES)
endif()

# Add subfolder with source files of the library
add_subdirectory(Src)
if(GEMPIC_USE_CUDA)
  # function defined in AMReX tools. No equivalent for HIP
  setup_target_for_cuda_compilation(GEMPICX)
endif()
# Add cmake-git-version-tracking
gempic_FetchContent_Declare(
  cmake_git_version_tracking
  GIT_REPOSITORY https://github.com/andrew-hardin/cmake-git-version-tracking.git
  GIT_TAG 6c0cb87edd029ddfb403a8e24577c144a03605a6
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/cmake_git_version_tracking-src
  )
set(GIT_WORKING_DIR ${PROJECT_SOURCE_DIR} CACHE STRING "Internal cmake_git_version_tracking option set within GEMPIC")
FetchContent_MakeAvailable(cmake_git_version_tracking)

target_link_libraries(GEMPICX
  PUBLIC AMReX::amrex_${AMReX_SPACEDIM}d AMReX::Flags_CXX
  # Private primarily so that people could include their own and get info about their repo
  # Info about our repo is already exposed in GEMPICX.H
  PRIVATE cmake_git_version_tracking)
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

gempic_install_export_targets()

# Preparations necessary for clang-tidy
include(cmake/clang_tidy.cmake)

# Config summary
message(STATUS "GEMPICX configuration summary: ")
message(STATUS "   AMReX version            = ${GEMPICX_AMReX_VERSION}")
message(STATUS "   GEMPICX version          = ${GEMPICX_GIT_VERSION}")
message(STATUS "   Build type               = ${CMAKE_BUILD_TYPE}")
message(STATUS "   Install directory        = ${CMAKE_INSTALL_PREFIX}")
message(STATUS "   C++ compiler             = ${CMAKE_CXX_COMPILER}")
if(${CMAKE_BUILD_TYPE} MATCHES "Release")
  message(STATUS "   C++ flags                = ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
elseif(${CMAKE_BUILD_TYPE} MATCHES "Debug")
  message(STATUS "   C++ flags                = ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
endif()
if(GEMPIC_USE_CUDA)
  message(STATUS "   CUDA compiler            = ${CMAKE_CUDA_COMPILER}")
  message(STATUS "   CUDA flags               = ${CMAKE_CUDA_FLAGS}")
endif()
if(GEMPIC_USE_HIP)
  message(STATUS "   HIP compiler            = ${CMAKE_HIP_COMPILER}")
  message(STATUS "   HIP flags               = ${CMAKE_HIP_FLAGS}")
endif()

gempic_create_and_link_cmake_constants_file()
endif() #CMAKE_BUILD_ONLY_DOCUMENTATION OFF
