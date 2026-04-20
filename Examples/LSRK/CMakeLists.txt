add_executable(LSRKvlasovmaxwell VlasovMaxwellSimulation.cpp)

if(GEMPIC_USE_CUDA) 
  setup_target_for_cuda_compilation(LSRKvlasovmaxwell) 
endif()

target_link_libraries(LSRKvlasovmaxwell cmake_git_version_tracking GEMPICX::GEMPICX)
