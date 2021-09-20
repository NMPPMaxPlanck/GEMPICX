add_executable ( test_gempic_norm test_gempic_norm.cpp)
add_executable ( test_AMReX_FillBoundary test_AMReX_FillBoundary.cpp )
add_executable ( test_AMReX_OverrideSync test_AMReX_OverrideSync.cpp )
add_executable ( test_AMReX_SumBoundary test_AMReX_SumBoundary.cpp )
add_executable ( test_AMReX_Dot test_AMReX_Dot.cpp )

#message("#########")
#message( ${AMReX_DIR} )
#message( " ######")
target_link_libraries( test_gempic_norm AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_AMReX_FillBoundary  AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_AMReX_OverrideSync  AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_AMReX_SumBoundary  AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_AMReX_Dot  AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )

if (USE_CUDA)
	setup_target_for_cuda_compilation( test_gempic_norm )
	setup_target_for_cuda_compilation( test_AMReX_FillBoundary )
	setup_target_for_cuda_compilation( test_AMReX_OverrideSync )
	setup_target_for_cuda_compilation( test_AMReX_SumBoundary )
	setup_target_for_cuda_compilation( test_AMReX_Dot )
endif()



######## Ctest

FILE(GLOB _tests *.expected_output)
LIST(SORT _tests)

FOREACH(_test ${_tests})
  GET_FILENAME_COMPONENT( _test ${_test} NAME_WE )

  _CTEST_FILE_CMP(${_test})

ENDFOREACH() 
