target_sources(GEMPICX PRIVATE
  GEMPIC_RungeKutta.cpp
  GEMPIC_Splitting.cpp)

add_library(GEMPICX_TIMESTEPPER)
target_sources(GEMPICX_TIMESTEPPER PRIVATE
               GEMPIC_TimeStepper.cpp)
target_include_directories(GEMPICX_TIMESTEPPER INTERFACE
                           ${CMAKE_CURRENT_LIST_DIR})

if(GEMPIC_USE_CUDA)
  # function defined in AMReX tools. No equivalent for HIP
  setup_target_for_cuda_compilation(GEMPICX_TIMESTEPPER)
endif()
target_link_libraries(GEMPICX_TIMESTEPPER PRIVATE AMReX::amrex_${AMReX_SPACEDIM}d AMReX::Flags_CXX)
target_link_libraries(GEMPICX INTERFACE GEMPICX_TIMESTEPPER)
