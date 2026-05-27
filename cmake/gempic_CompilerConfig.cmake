function(gempic_set_compile_options TARGET)
  message(STATUS "GEMPIC-CompilerID: ${CMAKE_CXX_COMPILER_ID}")
  string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
  if(BUILD_TYPE MATCHES "release")
    if(${CMAKE_CXX_COMPILER_ID} MATCHES "GNU.*")
      target_compile_options(${TARGET} PRIVATE -funroll-all-loops -march=native)
    endif()
  endif()
  if(BUILD_TYPE MATCHES "debug")
    target_compile_options(${TARGET} PRIVATE -Wall -Wextra)
    if(HDF5_VERSION MATCHES "1.14.1")
      # The following warning is removed to silence https://gitlab.mpcdf.mpg.de/gempic/gempic/-/jobs/4856938
      # The warning is triggered by the macro -D_FORTIFY_SOURCE=2 which propagates 
      # into GEMPICX through the HDF5 dependency specifically for the module hdf5-mpi/1.14.1.
      # For the latest HDF5 version this did not seem to be an issue.
      target_compile_options(${TARGET} PRIVATE "-Wno-#warnings")
    endif()
  endif()
  if(AMReX_SPACEDIM EQUAL 1)
    target_compile_options(${TARGET} PRIVATE -Wno-braced-scalar-init)
  endif()

  # NOTE: Enabling the link-time-optimization prolongs link times significantly!
  if(GEMPIC_USE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT IPO_IS_SUPPORTED OUTPUT IPO_ERROR_MSG)
    if(IPO_IS_SUPPORTED)
      message(STATUS "IPO/LTO enabled")
      set_target_properties(${TARGET} PROPERTY CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    else()
      message(STATUS "IPO/LTO not supported: ${IPO_ERROR_MSG}")
    endif()
  endif()
  if(GEMPIC_USE_HIP)
    target_compile_definitions(${TARGET} PRIVATE -DGEMPIC_GPU)
  endif()
  if(GEMPIC_USE_CUDA)
    target_compile_definitions(${TARGET} PRIVATE -DGEMPIC_GPU)
    target_compile_options(${TARGET} PRIVATE -lcusparse -lcurand)
    # The following flags are not found in the official nvcc documentation.
    # We found the usage here
    # https://stackoverflow.com/questions/14831051/how-to-disable-a-specific-nvcc-compiler-warnings
    # NVCC('s version of EDG?) is very trigger happy and does not respect [[maybe_unused]].
    # Warning tag name found at:
    # https://web.archive.org/web/20200605121301/www.ssl.berkeley.edu/~jimm/grizzly_docs/SSL/opt/intel/cc/9.0/lib/locale/en_US/mcpcom.msg
    target_compile_options(${TARGET} PRIVATE "SHELL:-Xcudafe --diag_suppress=set_but_not_used")
    target_compile_options(${TARGET} PRIVATE "SHELL:-Xcudafe --diag_suppress=declared_but_not_referenced")
  endif()
endfunction()

function(gempic_set_clang_toolchain_flags_on_mpcdf_systems)
  if(    (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang.*")
      OR (${CMAKE_CXX_COMPILER_ID} MATCHES "IntelLLVM.*"))
    if(   ($ENV{HOSTNAME} MATCHES "raven*")
       OR ($ENV{HOSTNAME} MATCHES "viper*")
       OR (DEFINED ENV{MPCDF_RUNNER}))
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gcc-toolchain=$ENV{GCC_HOME}" PARENT_SCOPE)
      set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON PARENT_SCOPE)
    endif()
  endif()
endfunction()

gempic_set_clang_toolchain_flags_on_mpcdf_systems()
