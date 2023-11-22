# Find the AMReX library
find_package( AMReX CONFIG )
if(NOT AMReX_FOUND)
    FetchContent_Declare(
    AMReX
    GIT_REPOSITORY https://github.com/AMReX-Codes/amrex.git
    GIT_TAG 23.09
    )

    set( AMReX_PARTICLES ON CACHE BOOL "AMReX Option set within GEMPIC")
    if( GEMPIC_USE_CUDA )
        set( AMReX_GPU_BACKEND CUDA CACHE STRING "AMReX Option set within GEMPIC")
        if( $ENV{HOST} MATCHES "raven")
          set( AMReX_CUDA_ARCH "Ampere" CACHE STRING "AMReX Option set within GEMPIC")
        endif()
    endif()
    if( GEMPIC_USE_OMP ) 
        set( AMReX_OMP  ON CACHE BOOL "AMReX Option set within GEMPIC")
    else()
        set( AMReX_OMP OFF CACHE BOOL "AMReX Option set within GEMPIC")
    endif()

    FetchContent_MakeAvailable(AMReX)
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
    set_source_files_properties(${_sources} PROPERTIES LANGUAGE CUDA )
  endfunction()

  #
  # Setup an amrex-dependent target for cuda compilation.
  # This function ensures that the CUDA compilation of _target
  # is compatible with amrex CUDA build.
  #
  function(Setup_target_for_cuda_compilation _target)
    set_target_properties( ${_target}
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON      # This adds -dc
        )
    set_cpp_sources_to_cuda_language(${_target})
  endfunction()
  
endif(GEMPIC_USE_CUDA)

