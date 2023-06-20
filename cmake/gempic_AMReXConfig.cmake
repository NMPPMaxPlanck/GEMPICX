#find_package(AMReX CONFIG)
message("AMReX Dir is ${AMReX_DIR}")

# Find the AMReX library
find_package( AMReX  HINTS ${AMReX_DIR}/lib/cmake/AMReX )
if(NOT AMReX_FOUND)
    #FetchContent_Declare(
    #AMReX
    #GIT_REPOSITORY https://github.com/AMReX-Codes/amrex.git
    ## Latest commit before particle class was changed. Fetching newer versions will require us to change particle treatment in quite a lot of places.
    #GIT_TAG 5bbb63f24753353bdcbb8c439fedbc3dfc11d15e
    #)

    set( AMReX_PARTICLES ON )
    if( GEMPIC_USE_CUDA )
        set( AMReX_GPU_BACKEND CUDA )
    endif()
    add_subdirectory( third_party/amrex )
    if( GEMPIC_USE_OMP ) 
        set( AMReX_OMP "YES" )
        set( AMReX_OMP ON )
    else()
        set( AMReX_OMP OFF )
    endif()

    #FetchContent_MakeAvailable(AMReX)
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

