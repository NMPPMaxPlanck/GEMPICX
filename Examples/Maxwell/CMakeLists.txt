add_executable(maxwell Maxwell.cpp GEMPIC_MaxwellInit.cpp)

if(GEMPIC_USE_CUDA) 
  setup_target_for_cuda_compilation(maxwell) 
endif()

target_link_libraries(maxwell GEMPICX::GEMPICX)
