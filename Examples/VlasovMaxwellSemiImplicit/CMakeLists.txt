add_executable(vlasovmaxwellsemiimp VlasovMaxwellSemiImplicit.cpp)

if(GEMPIC_USE_CUDA) 
  setup_target_for_cuda_compilation(vlasovmaxwellsemiimp) 
endif()

target_link_libraries(vlasovmaxwellsemiimp GEMPICX)

