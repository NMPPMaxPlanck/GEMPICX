# Minimum version of CMake required to build this project
cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 14)
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Name of the project
project("gempic")

option(USE_CUDA "Use CUDA" OFF)
option(USE_OMP "Use OpenMP" OFF)

# Enable Languages
enable_language(CXX)
if (USE_CUDA)
  enable_language(CUDA)
  set(CUDA_SEPARABLE_COMPILATION ON)
  find_package(CUDA QUIET REQUIRED)
endif()

if (USE_OMP)
  find_package(OpenMP REQUIRED)
  if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    set (AMReX_OMP "YES")
  endif()
endif()

find_package(MPI REQUIRED CXX)

message("*****MPI path ${MPI_CUDA_HEADER_DIR}, ${MPI_CXX_HEADER_DIR}")
# Set compiler flags

message("####### ${CMAKE_CXX_COMPILER_ID}")
IF (CMAKE_CXX_COMPILER_ID MATCHES "GNU.*")
  # gcc
  #set (CMAKE_CXX_FLAGS "")
  set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -funroll-all-loops -O3 -mavx2  -march=native")
  set (CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -Wall")
  message ("Found GNU compiler " ${CMAKE_CXX_COMPILER})
ELSEIF (CMAKE_CXX_COMPILER_ID MATCHES "Intel.*")
	#set (CMAKE_CXX_FLAGS "")
	set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -funroll-all-loops -O3 -xHost -zmm-usage=high -Ofast -qopt-prefetch=2 -mcmodel=large") # or try 4 for prefetch
  #set (CMAKE_CXX_FLAGS_RELEASE "-g -O3 -xHOST")
  set (CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -Wall")
  message ("Found Intel compiler " ${CMAKE_CXX_COMPILER})
ELSEIF (CMAKE_CXX_COMPILER_ID MATCHES "Clang.*")
  # Clang 
  set (CMAKE_CXX_FLAGS "")
  set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -O3")
  set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -Wall -Wextra")
  message ("Found Clang compiler " ${CMAKE_CXX_COMPILER})  
ELSE()
	#set (CMAKE_CXX_FLAGS "")
	#set (CMAKE_CXX_FLAGS_RELEASE "")
  set (CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -Wall")
  message("Unknown compiler")
ENDIF()

IF (USE_CUDA)
  add_definitions(-DGEMPIC_GPU)
  IF (($ENV{HOST} MATCHES "cobra*") OR ($ENV{HOST} MATCHES "raven*"))
    IF (CMAKE_CXX_COMPILER_ID MATCHES "Intel.*")  
	    set (CMAKE_CUDA_FLAGS "--Werror cross-execution-space-call --expt-extended-lambda --expt-relaxed-constexpr -ccbin=mpiicpc -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64 -lcurand")
      set (CMAKE_CUDA_FLAGS_RELEASE "--Werror cross-execution-space-call --expt-extended-lambda --expt-relaxed-constexpr -ccbin=mpiicpc -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64 -lcurand -arch sm_80")
      set (CMAKE_CUDA_FLAGS_DEBUG   "--Werror cross-execution-space-call --expt-extended-lambda --expt-relaxed-constexpr -ccbin=mpiicpc -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64 -lcurand -arch sm_80")
    ENDIF()
  ENDIF()
  IF (($ENV{HOST} MATCHES "daint*") )
    set (CMAKE_CUDA_FLAGS "--Werror cross-execution-space-call --expt-extended-lambda --expt-relaxed-constexpr -ccbin=CC -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64 -lcurand  -arch sm_60")
    set (CMAKE_CUDA_FLAGS_RELEASE "--Werror cross-execution-space-call --expt-extended-lambda --expt-relaxed-constexpr -ccbin=CC -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64 -lcurand -arch sm_60")
    set (CMAKE_CUDA_FLAGS_DEBUG   "--Werror cross-execution-space-call --expt-extended-lambda --expt-relaxed-constexpr -ccbin=CC -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64 -lcurand  -arch sm_60")
  ENDIF()

  find_package(CUDAToolkit REQUIRED)
  set (CMAKE_CUDA_FLAGS "-lcusparse -lcurand" )
ENDIF()

message("AMReX Dir is ${AMReX_DIR}")

# Find the AMReX library
find_package( AMReX  HINTS ${AMReX_DIR}/lib/cmake/AMReX )
if (NOT AMReX_FOUND)
#  set( AMReX_SPACEDIM 3 )
  set( AMReX_PARTICLES ON )
  # set( AMReX_TINY_PROFILE ON )
  if ( USE_CUDA )
    set( AMReX_GPU_BACKEND CUDA )
  endif()
  add_subdirectory( third_party/amrex )
  if ( USE_OMP ) 
    set ( AMReX_OMP ON )
  else()
    set( AMReX_OMP OFF )
  endif()
endif()
# This is an old version to link with amrex (gfortran and intel versions, respectively
#set( AMReX_LIBRARIES ${AMReX_ROOT}/lib/libamrex.a;gfortran )
#set( AMReX_LIBRARIES ${AMReX_ROOT}/lib/libamrex.a;ifcore )
#include_directories( ${AMReX_DIR}/include )
#message("AMReX_LIBRARIES ${AMReX_LIBRARIES}")
#message("AMReX Dir is ${AMReX_DIR} ${AMReX_ROOT}")
#set( AMREX ON CACHE BOOL "AMREX found." )

#set (CMAKE_CUDAA_FLAGS_RELEASE AMReX::Flags_CXX)

# Enable ctests
enable_testing()
# This defines how the ctests are run
message ("HOST is $ENV{HOST}" )
IF (($ENV{HOST} MATCHES "cobra*") OR ($ENV{HOST} MATCHES "raven*") OR ($ENV{HOST} MATCHES "daint*") )
  include(cmake/cobra_ctest.cmake)
ELSE()
  include(cmake/define_ctest.cmake)
ENDIF()

if (USE_CUDA)

  # Convert all .cpp sources of _target to CUDA sources
  # This DOES NOT change the actual extension of the source.
  # It just change the default language CMake uses to compile
  # the source
  #
  function (set_cpp_sources_to_cuda_language _target)
    get_target_property(_sources ${_target} SOURCES)
    list(FILTER _sources INCLUDE REGEX "\\.cpp$")
    set_source_files_properties(${_sources} PROPERTIES LANGUAGE CUDA )
  endfunction ()

  #
  # Setup an amrex-dependent target for cuda compilation.
  # This function ensures that the CUDA compilation of _target
  # is compatible with amrex CUDA build.
  #
  function (setup_target_for_cuda_compilation _target)
	  set_target_properties( ${_target}
	  PROPERTIES
	  CUDA_SEPARABLE_COMPILATION ON      # This adds -dc
	   )
   set_cpp_sources_to_cuda_language(${_target})

   #   if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.20)
   #   set_target_properties( ${_target}
   #      PROPERTIES
   #      CUDA_ARCHITECTURES "${AMREX_CUDA_ARCHS}"
   #      )
   #endif ()
  endfunction ()

endif()

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
  message( STATUS "   C++ flags                = ${CMAKE_CXX_FLAGS_DEBUG}")
endif()
if (USE_CUDA)
message( STATUS "   CUDA compiler            = ${CMAKE_CUDA_COMPILER}")
message( STATUS "   CUDA flags               = ${CMAKE_CUDA_FLAGS}")
endif()

