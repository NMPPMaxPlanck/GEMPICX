add_executable ( test_checkpoint test_checkpoint.cpp )

if (USE_CUDA) 
	setup_target_for_cuda_compilation( test_checkpoint )
endif()

target_link_libraries( test_checkpoint  AMReX::amrex AMReX::Flags_CXX  )


######## Ctest

FILE(GLOB _tests *.expected_output)
LIST(SORT _tests)

FOREACH(_test ${_tests})
  GET_FILENAME_COMPONENT( _test ${_test} NAME_WE )

  _CTEST_FILE_CMP(${_test})

ENDFOREACH()
