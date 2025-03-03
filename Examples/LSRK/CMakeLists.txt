add_executable(LSRKvlasovmaxwell VlasovMaxwellSimulation.cpp)

if(GEMPIC_USE_CUDA) 
  setup_target_for_cuda_compilation(LSRKvlasovmaxwell) 
endif()

target_link_libraries(LSRKvlasovmaxwell GEMPICX cmake_git_version_tracking)

