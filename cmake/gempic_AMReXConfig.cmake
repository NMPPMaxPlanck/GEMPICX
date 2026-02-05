macro(set_amrex_options_from_gempic)
  include(cmake/check_FFT.cmake)
  set(AMReX_PARTICLES ON CACHE BOOL "AMReX Option set within GEMPIC")
  if(GEMPIC_USE_HYPRE)
    set(AMReX_HYPRE ON CACHE BOOL "AMReX Option set by default within GEMPIC")
  endif()
  if(GEMPIC_USE_CUDA)
    if(IPO_IS_SUPPORTED)
      set(AMReX_CUDA_LTO ON CACHE STRING "AMReX Option set within GEMPIC")
    endif()
    set(AMReX_GPU_BACKEND CUDA CACHE STRING "AMReX Option set within GEMPIC")
  endif()
  if(GEMPIC_USE_HIP)
    set(AMReX_GPU_BACKEND HIP CACHE STRING "AMReX Option set within GEMPIC")
    set(AMReX_AMD_ARCH ${CMAKE_HIP_ARCHITECTURES} CACHE STRING "AMReX Option set within GEMPIC")
  endif()
  if(GEMPIC_USE_OMP) 
    set(AMReX_OMP  ON CACHE BOOL "AMReX Option set within GEMPIC")
  else()
    set(AMReX_OMP OFF CACHE BOOL "AMReX Option set within GEMPIC")
  endif()
endmacro()

include(cmake/gempic_FetchContent_Declare.cmake)

if(GEMPIC_USE_HYPRE)
  gempic_FetchContent_Declare(HYPRE
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/hypre-src
    GIT_REPOSITORY https://github.com/hypre-space/hypre
    GIT_TAG v2.32.0
    SOURCE_SUBDIR src # hypre doesn't follow standard cmake conventions
    OVERRIDE_FIND_PACKAGE
  )
endif()
gempic_FetchContent_Declare(AMReX
             SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/amrex-src
             GIT_REPOSITORY https://github.com/AMReX-Codes/amrex.git
             #GIT_TAG 5686ee3 # AMReX commit Nov 4, 2025
             GIT_TAG 25.11
             ALLOW_DIRTY ${USE_DIRTY_AMREX_REPO}
             GIT_PROGRESS ON # AMReX takes long enough that this is nice instead of noise.
             )
if(NOT ${AMReX_FOUND}) # AMReX_FOUND is only true if the package was installed
  set_amrex_options_from_gempic() # and only if not do the settings matter.
  FetchContent_MakeAvailable(AMReX)
  set(AMReX_VERSION_USED ${AMReX_GIT_TAG})
else()
  set(AMReX_VERSION_USED ${AMReX_VERSION})
endif()

if(GEMPIC_USE_CUDA)
  # Convert all .cpp sources of _target to CUDA sources
  # This DOES NOT change the actual extension of the source.
  # It just change the default language CMake uses to compile
  # the source
  #
  function(set_cpp_sources_to_cuda_language _target)
    get_target_property(_sources ${_target} SOURCES)
    list(FILTER _sources INCLUDE REGEX "\\.cpp$")
    set_source_files_properties(${_sources} PROPERTIES LANGUAGE CUDA)
  endfunction()

  #
  # Setup an amrex-dependent target for cuda compilation.
  # This function ensures that the CUDA compilation of _target
  # is compatible with amrex CUDA build.
  #
  function(Setup_target_for_cuda_compilation _target)
    set_target_properties(${_target}
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON      # This adds -dc
        )
    set_cpp_sources_to_cuda_language(${_target})
  endfunction()
  
endif(GEMPIC_USE_CUDA)
