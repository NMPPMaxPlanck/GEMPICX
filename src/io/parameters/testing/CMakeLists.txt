add_executable ( test_parser test_parser.cpp )

if (USE_CUDA) 
	setup_target_for_cuda_compilation( test_parser )
endif()

target_link_libraries( test_parser AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )


######## Ctest

FILE(GLOB _tests *.expected_output)
LIST(SORT _tests)

FOREACH(_test ${_tests})
  GET_FILENAME_COMPONENT( _test ${_test} NAME_WE )

  _CTEST_FILE_CMP(${_test})

ENDFOREACH()
