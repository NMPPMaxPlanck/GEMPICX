add_executable(electrostatic ElectrostaticSimulation.cpp)

if(GEMPIC_USE_CUDA) 
  setup_target_for_cuda_compilation(electrostatic) 
endif()

target_link_libraries(electrostatic GEMPICX)

