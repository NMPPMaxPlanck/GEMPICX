add_executable(dkquasineutralvm DKQuasineutralVMSimulation.cpp)

if(GEMPIC_USE_CUDA)
  setup_target_for_cuda_compilation(dkquasineutralvm)
endif()

target_link_libraries(dkquasineutralvm GEMPICX)

