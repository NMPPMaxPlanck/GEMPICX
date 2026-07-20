macro(set_kokkos_options_from_gempic)
  if(GEMPIC_USE_CUDA)
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Kokkos Option set within GEMPIC")
    set(Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE ON CACHE BOOL "Kokkos Option set within GEMPIC")
    set(Kokkos_ENABLE_CONSTEXPR ON CACHE BOOL "Kokkos Option set within GEMPIC")
    if(($ENV{HOST} MATCHES "raven") OR
       (($ENV{CI_RUNNER_TAGS} MATCHES "mpcdf-shared") AND ($ENV{CI_RUNNER_TAGS} MATCHES "gpu-nvidia")))
      set(Kokkos_ARCH_AMPERE80 ON CACHE BOOL "Kokkos Option set within GEMPIC")
      # Setting the CUDA_ARCHITECTURE explicitly is required to propagate the
      # architecture to AMReX as well.
      set(CMAKE_CUDA_ARCHITECTURES 80)
    endif()
  endif()
  if(GEMPIC_USE_HIP)
    set(Kokkos_ENABLE_HIP CACHE ON "Kokkos Option set within GEMPIC")
    set(KOKKOS_ARCH_AMD_GFX942 ON CACHE STRING "Kokkos Option set within GEMPIC")
  endif()
  if(GEMPIC_USE_OMP)
    set(Kokkos_ENABLE_OPENMP  ON CACHE BOOL "Kokkos Option set within GEMPIC")
  else()
    set(Kokkos_ENABLE_OPENMP OFF CACHE BOOL "Kokkos Option set within GEMPIC")
  endif()
endmacro()

include(cmake/gempic_FetchContent_Declare.cmake)

gempic_FetchContent_Declare(Kokkos
             SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/kokkos-src
             GIT_REPOSITORY https://github.com/kokkos/kokkos
             GIT_TAG 297e182502896611ccc699b4935d1bcc703293c9 # Kokkos release 5.0.1
             ALLOW_DIRTY OFF
             GIT_PROGRESS ON
             )
if(NOT ${Kokkos_FOUND})
  set_kokkos_options_from_gempic()
  FetchContent_MakeAvailable(Kokkos)
endif()

