add_executable(lin_vlasov_maxwell LinVlasovMaxwellSimulation.cpp)

if(GEMPIC_USE_CUDA) 
  setup_target_for_cuda_compilation(lin_vlasov_maxwell) 
endif()

target_link_libraries(lin_vlasov_maxwell GEMPICX cmake_git_version_tracking)

