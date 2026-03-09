message(STATUS "GEMPIC-CompilerID: ${CMAKE_CXX_COMPILER_ID}")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU.*")
  # gcc
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -funroll-all-loops -march=native")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
  message(STATUS "GEMPIC-Compiler: GNU")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel.*")
  # Intel compilers, new LLVM based versions (icx, icpx) as well as the legacy ones (icc, icpc) work well with these flags.
  # Note that AVX512 is available on virtually all recent HPC-class x86_64 processors but not on Laptops and Desktops.
  # Native optimization with the Intel compiler on AMD CPUs causes issues, so we set skylake-avx512 explicitly here:
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=skylake-avx512")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
  message(STATUS "GEMPIC-Compiler: Intel")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang.*")
  # Clang 
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra")
  if(AMReX_SPACEDIM EQUAL 1)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wno-braced-scalar-init")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wno-braced-scalar-init")
  endif()
  # MPCDF environments only provide full c++ std support for GCC.
  # Point the clang implementation to the correct toolchain
  if(   ($ENV{HOSTNAME} MATCHES "raven*")
     OR ($ENV{HOSTNAME} MATCHES "viper*")
     OR (DEFINED ENV{MPCDF_RUNNER}))
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} --gcc-toolchain=$ENV{GCC_HOME}")
    set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} --gcc-toolchain=$ENV{GCC_HOME}")
  endif()
  message(STATUS "GEMPIC-Compiler: Clang")
else()
  message(STATUS "GEMPIC-Compiler: Unknown compiler")
endif()


# NOTE: Enabling the link-time-optimization prolongs link times significantly!
if(GEMPIC_USE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT IPO_IS_SUPPORTED OUTPUT IPO_ERROR_MSG)
  if(IPO_IS_SUPPORTED)
    message(STATUS "IPO/LTO enabled")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE) # All targets instantiated afterwards
  else()
    message(STATUS "IPO/LTO not supported: ${IPO_ERROR_MSG}")
  endif()
endif()


if(GEMPIC_USE_CUDA)
  add_definitions(-DGEMPIC_GPU)
  # Do we really only compile for GPU on raven?
  # Flags should probably revisited
  if($ENV{HOST} MATCHES "raven*")
    # Are these flags reached at all? We do no compile GPU code with intel compiler
    if(CMAKE_CXX_COMPILER_ID MATCHES "Intel.*")  
      set(CMAKE_CUDA_FLAGS "--Werror cross-execution-space-call --expt-extended-lambda --expt-relaxed-constexpr -ccbin=mpiicpc -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64 -lcurand")
      set(CMAKE_CUDA_FLAGS_RELEASE "--Werror cross-execution-space-call --expt-extended-lambda --expt-relaxed-constexpr -ccbin=mpiicpc -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64 -lcurand -arch sm_80")
      set(CMAKE_CUDA_FLAGS_DEBUG   "--Werror cross-execution-space-call --expt-extended-lambda --expt-relaxed-constexpr -ccbin=mpiicpc -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64 -lcurand -arch sm_80")
    endif()
  endif()

  find_package(CUDAToolkit REQUIRED)
  # NVCC('s version of EDG?) is very trigger happy and does not respect [[maybe_unused]].
  # Warning tag name found at:
  # http://www.ssl.berkeley.edu/~jimm/grizzly_docs/SSL/opt/intel/cc/9.0/lib/locale/en_US/mcpcom.msg
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lcusparse -lcurand -Xcudafe --diag_suppress=set_but_not_used -Xcudafe --diag_suppress=declared_but_not_referenced" )

  # This doesn't do anything but avoid an AMReX warning -- CMAKE_CUDA_HOST_COMPILER must be specified as a -D option on the first invocation of cmake as it is used during the compiler detection process.
  if (CMAKE_CUDA_HOST_COMPILER)
    if ("${CMAKE_CXX_COMPILER}" MATCHES "${CMAKE_CUDA_HOST_COMPILER}")
      set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    endif()
  endif()
endif()
