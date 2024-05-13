add_executable(test_parameters test_parameters.cpp)

if(GEMPIC_USE_CUDA) 
  setup_target_for_cuda_compilation(test_parameters)
endif()

target_link_libraries(test_parameters GEMPICX)

list(SORT _tests)

if("${AMReX_SPACEDIM}" STREQUAL "3")
  add_subdirectory(IOFiles3D)
endif()
