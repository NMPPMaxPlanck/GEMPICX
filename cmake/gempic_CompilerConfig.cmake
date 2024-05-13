message(STATUS "GEMPIC-CompilerID: ${CMAKE_CXX_COMPILER_ID}")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU.*")
  # gcc
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -funroll-all-loops -mavx2  -march=native")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
  message(STATUS "GEMPIC-Compiler: GNU")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel.*")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -funroll-all-loops -xHost -zmm-usage=high -Ofast -qopt-prefetch=2 -mcmodel=large") # or try 4 for prefetch
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
  message(STATUS "GEMPIC-Compiler: Intel")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang.*")
  # Clang 
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra")
  if(AMReX_SPACEDIM EQUAL 1)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wno-braced-scalar-init")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wno-braced-scalar-init")
  endif()
  message(STATUS "GEMPIC-Compiler: Clang")
else()
  message(STATUS "GEMPIC-Compiler: Unknown compiler")
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_IS_SUPPORTED OUTPUT IPO_ERROR_MSG)
if(IPO_IS_SUPPORTED)
  message(STATUS "IPO/LTO enabled")
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE) # All targets instantiated afterwards
else()
  message(STATUS "IPO/LTO not supported: ${IPO_ERROR_MSG}")
endif()


if(GEMPIC_USE_CUDA)
  add_definitions(-DGEMPIC_GPU)
  if(($ENV{HOST} MATCHES "cobra*") OR ($ENV{HOST} MATCHES "raven*"))
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
  set(CMAKE_CUDA_FLAGS "-lcusparse -lcurand -Xcudafe --diag_suppress=set_but_not_used" )


  # This doesn't do anything but avoid an AMReX warning -- CMAKE_CUDA_HOST_COMPILER must be specified as a -D option on the first invocation of cmake as it is used during the compiler detection process.
  if (CMAKE_CUDA_HOST_COMPILER)
    if ("${CMAKE_CXX_COMPILER}" MATCHES "${CMAKE_CUDA_HOST_COMPILER}")
      set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    endif()
  endif()
endif()
