target_sources(GEMPICX PRIVATE GEMPIC_Splitting.cpp GEMPIC_RungeKutta.cpp)

add_library(
  GEMPICX_TIMESTEPPER
  GEMPIC_TimeStepper.cpp)

target_include_directories(
  GEMPICX_TIMESTEPPER
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)

if(GEMPIC_USE_CUDA)
  # function defined in AMReX tools. No equivalent for HIP
  setup_target_for_cuda_compilation(GEMPICX_TIMESTEPPER)
endif()
target_link_libraries(GEMPICX_TIMESTEPPER PUBLIC AMReX::amrex_${AMReX_SPACEDIM}d AMReX::Flags_CXX)
