add_executable(quasi_neutral QuasiNeutralSimulation.cpp)

if(GEMPIC_USE_CUDA) 
  setup_target_for_cuda_compilation(quasi_neutral) 
endif()

target_link_libraries(quasi_neutral GEMPICX)

