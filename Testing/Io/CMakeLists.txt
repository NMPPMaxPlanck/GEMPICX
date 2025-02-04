target_sources(GtestTests PRIVATE
  # test cases
  Parameters_test.cpp
  FullDiagnostics_test.cpp
  ReducedDiagnostics_test.cpp
  )

add_executable(ParametersOutput_test ParametersOutput_test.cpp)

if(GEMPIC_USE_CUDA) 
  setup_target_for_cuda_compilation(ParametersOutput_test)
endif()

target_link_libraries(ParametersOutput_test GEMPICX)

add_test(NAME ParametersOutput_test
  COMMAND ${CMAKE_COMMAND} 
    -DPARAMETERS_OUTPUT_TEST=$<TARGET_FILE:ParametersOutput_test>
    -DINPUT_FILE=${CMAKE_SOURCE_DIR}/Testing/InputFiles/ParametersOutput_test.input
  -P ${CMAKE_CURRENT_SOURCE_DIR}/ParametersOutput_test.cmake
  )
