add_executable(vlasovmaxwell HSVlasovMaxwell.cpp)

if(GEMPIC_USE_CUDA) 
  setup_target_for_cuda_compilation(vlasovmaxwell) 
endif()

target_link_libraries(vlasovmaxwell GEMPICX)

