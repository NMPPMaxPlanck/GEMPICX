add_executable(cold_plasma ColdPlasmaSimulation.cpp)

if(GEMPIC_USE_CUDA) 
  setup_target_for_cuda_compilation(cold_plasma) 
endif()

target_link_libraries(cold_plasma GEMPICX)