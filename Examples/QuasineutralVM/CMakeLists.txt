add_executable(quasineutralvm QuasineutralVMSimulation.cpp)

if(GEMPIC_USE_CUDA)
  setup_target_for_cuda_compilation(quasineutralvm)
endif()

target_link_libraries(quasineutralvm GEMPICX)

