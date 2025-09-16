add_executable(MHD MHD.cpp)

if(MPIDEBUG)
    target_compile_definitions(MHD PRIVATE MPIDEBUG)
endif()

if(GEMPIC_USE_CUDA) 
  setup_target_for_cuda_compilation(MHD) 
endif()

target_link_libraries(MHD GEMPICX)
