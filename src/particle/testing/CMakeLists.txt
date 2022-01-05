add_executable ( test_sampler test_sampler.cpp )

if (USE_CUDA) 
	setup_target_for_cuda_compilation ( test_sampler ) 
endif()

#message( ${AMReX_DIR} )
#message( " ######")
target_link_libraries( test_sampler  AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran  )

######## Ctest

FILE(GLOB _tests *.expected_output)
LIST(SORT _tests)

FOREACH(_test ${_tests})
  GET_FILENAME_COMPONENT( _test ${_test} NAME_WE )

  _CTEST_FILE_CMP(${_test})

ENDFOREACH()
