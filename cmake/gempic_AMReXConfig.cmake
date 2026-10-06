macro(set_amrex_options_from_gempic)
  include(cmake/check_FFT.cmake)
  set(AMReX_PARTICLES ON CACHE BOOL "AMReX Option set within GEMPIC")
  # Some parts of fetch_content(HDF5) trigger AMReX shared libraries.
  # This line configures them to be disabled
  # ToDo: Shared libraries should not depend on the HDF5 build.
  #       Further investigation required
  set(AMReX_BUILD_SHARED_LIBS OFF CACHE BOOL "AMReX Option set within GEMPIC")
  # HDF5 is searched/fetched after AMReX is configured/searched.
  # Set it to off to avoid interference
  set(AMReX_HDF5              OFF CACHE BOOL "AMReX Option set within GEMPIC")
  if(GEMPIC_USE_CUDA)
    # AMReX does not recognise the CUDA language set by Kokkos, so we still need to enable it
    # even if Kokkos handles the performance portability options
    enable_language(CUDA)
    set(AMReX_CUDA_LTO ON CACHE STRING "AMReX Option set within GEMPIC")
    set(AMReX_GPU_BACKEND CUDA CACHE STRING "AMReX Option set within GEMPIC")
  endif()
  if(GEMPIC_USE_HIP)
    enable_language(HIP)
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

if(AMReX_HYPRE)
  gempic_FetchContent_Declare(HYPRE
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/hypre-src
    GIT_REPOSITORY https://github.com/hypre-space/hypre
    GIT_TAG v2.32.0
    SOURCE_SUBDIR src # hypre doesn't follow standard cmake conventions
    OVERRIDE_FIND_PACKAGE
  )
endif()
gempic_FetchContent_Declare(AMReX
             SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/amrex-src
             GIT_REPOSITORY https://github.com/AMReX-Codes/amrex.git
             # AMReX commit Apr, 2026 to fix --prefix option bug
             GIT_TAG ac6ea0c009071599b873d3b1296b39955d3faa1c
             ALLOW_DIRTY ${USE_DIRTY_AMREX_REPO}
             GIT_PROGRESS ON # AMReX takes long enough that this is nice instead of noise.
             )
if(NOT ${AMReX_FOUND}) # AMReX_FOUND is only true if the package was installed
  set_amrex_options_from_gempic() # and only if not do the settings matter.
  FetchContent_MakeAvailable(AMReX)
  if(GEMPIC_USE_CUDA)
    get_target_property(_amrex_ico amrex_${AMReX_SPACEDIM}d INTERFACE_COMPILE_OPTIONS)
    set_target_properties(amrex_${AMReX_SPACEDIM}d PROPERTIES INTERFACE_COMPILE_OPTIONS
      "SHELL:-Xcudafe --diag_suppress=20012;${_amrex_ico}")
  endif()
  if(AMReX_HYPRE)
    gempic_suppress_third_party_warnings(TARGET HYPRE)
  endif()
endif()
