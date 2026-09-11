# ------------------------------------------------------------
# gempic_LapackConfig.cmake
# Handles LAPACK / MKL configuration for GEMPIC
# ------------------------------------------------------------

# Only proceed if LAPACK or MKL is requested
if(NOT GEMPIC_USE_LAPACK AND NOT GEMPIC_USE_MKL)
    return()
endif()

# currently we don't support both MKL and LAPACK at the same time, so error out if both are enabled
if(GEMPIC_USE_MKL AND GEMPIC_USE_LAPACK)
  message(FATAL_ERROR "Cannot enable both MKL and LAPACK")
endif()

include(cmake/gempic_FetchContent_Declare.cmake)

# -----------------------------------------------------------------------------
# Option 1: Use Intel MKL
# -----------------------------------------------------------------------------
if(GEMPIC_USE_MKL)
  find_package(MPI REQUIRED)
  execute_process(
    COMMAND ${MPIEXEC_EXECUTABLE} --version
    OUTPUT_VARIABLE MPIEXEC_VERSION_OUT
    ERROR_VARIABLE MPIEXEC_VERSION_ERR)
  if("${MPIEXEC_VERSION_OUT}" MATCHES "open-mpi")
    # message(STATUS "OpenMPI found: ${MPIEXEC_VERSION_OUT}")
    set(MKL_MPI "openmpi")
  elseif("${MPIEXEC_VERSION_OUT}" MATCHES "Intel")
    # message(STATUS "Intel MPI found: ${MPIEXEC_VERSION_OUT}")
    set(MKL_MPI "intelmpi")
  else()
    # message(STATUS "Unknown\n${MPIEXEC_VERSION_OUT}\ntrying Windows")
    execute_process(
      COMMAND ${MPIEXEC_EXECUTABLE} -help
      OUTPUT_VARIABLE MPIEXEC_HELP_OUT
      ERROR_VARIABLE MPIEXEC_HELP_ERR)
    if("${MPIEXEC_HELP_OUT}" MATCHES "Microsoft MPI")
      # message(STATUS "MSMPI found: ${MPIEXEC_HELP_OUT}")
      set(MKL_MPI "msmpi")
    else()
      message(FATAL_ERROR "Unknown MPI version ${MPIEXEC_EXECUTABLE}")
    endif()
  endif()
    find_package(MKL REQUIRED)
    if(NOT TARGET MKL::MKL)
        message(FATAL_ERROR "MKL target not found!")
    endif()
    # Provide a unified interface target
    add_library(gempic_linalg INTERFACE)
    target_link_libraries(gempic_linalg INTERFACE MKL::MKL)
    set_target_properties(gempic_linalg PROPERTIES FOLDER "GEMPIC")
    return()
endif()

# -----------------------------------------------------------------------------
# Option 2: Use Netlib LAPACK / LAPACKE
# -----------------------------------------------------------------------------
if(GEMPIC_USE_LAPACK)

    # Set LAPACKE build options
    set(LAPACKE ON CACHE BOOL "" FORCE)
    set(LAPACK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    
    # Declare LAPACK via FetchContent
    gempic_FetchContent_Declare(
        LAPACK
        GIT_REPOSITORY https://github.com/Reference-LAPACK/lapack.git
        GIT_TAG 05a6d9f4e9004977b6ecc045ec1c42756134468e
        ALLOW_DIRTY ${USE_DIRTY_LAPACK_REPO}
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/lapack-src
        GIT_PROGRESS ON
    )

    FetchContent_MakeAvailable(LAPACK)
    add_library(gempic_linalg INTERFACE)
    set_target_properties(gempic_linalg PROPERTIES FOLDER "GEMPIC")
    
    if(TARGET lapacke)
        target_link_libraries(gempic_linalg INTERFACE lapacke)
    else()
        message(FATAL_ERROR "LAPACKE target not found")
    endif()

    # Link LAPACK / LAPACKE / BLAS
    if(TARGET LAPACK::LAPACK)
        target_link_libraries(gempic_linalg INTERFACE
            LAPACK::LAPACK
            lapacke
            BLAS::BLAS
        )
    else()
        # Fallback for older versions / raw library names
        target_link_libraries(gempic_linalg INTERFACE
            lapack
            lapacke
            blas
        )
    endif()
endif()

if(GEMPIC_USE_MKL)
    target_compile_definitions(gempic_linalg INTERFACE GEMPIC_USE_MKL=1)
endif()

if(GEMPIC_USE_LAPACK)
    target_compile_definitions(gempic_linalg INTERFACE GEMPIC_USE_LAPACK=1)
endif()