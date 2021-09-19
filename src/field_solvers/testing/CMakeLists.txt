add_executable ( test_maxwell_yee test_maxwell_yee.cpp )
add_executable ( test_maxwell_yee_order test_maxwell_yee_order.cpp )
add_executable ( test_poisson_CG test_poisson_CG.cpp )
add_executable ( test_ihodge_CG test_ihodge_CG.cpp )
add_executable ( test_ampere_faraday test_ampere_faraday.cpp )
add_executable ( test_AMReX_atomic_add test_AMReX_atomic_add.cpp )
add_executable ( test_constval test_constval.cpp )

if (USE_CUDA)
	setup_target_for_cuda_compilation( test_maxwell_yee )
	setup_target_for_cuda_compilation( test_maxwell_yee_order )
	setup_target_for_cuda_compilation( test_poisson_CG )
	setup_target_for_cuda_compilation( test_ihodge_CG )
	setup_target_for_cuda_compilation( test_ampere_faraday )
	setup_target_for_cuda_compilation( test_AMReX_atomic_add )
	setup_target_for_cuda_compilation( test_constval )
endif()


target_link_libraries( test_maxwell_yee gempic_poisson AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_maxwell_yee_order gempic_poisson AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_poisson_CG gempic_poisson AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_ihodge_CG gempic_poisson AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_ampere_faraday gempic_poisson AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_AMReX_atomic_add gempic_poisson AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_constval AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )

#target_link_libraries( test_FillBD ${AMReX_LIBRARIES} )


if (USE_CUDA)
  setup_target_for_cuda_compilation( test_constval )
endif()

######## Ctest

FILE(GLOB _tests *.expected_output)
LIST(SORT _tests)

FOREACH(_test ${_tests})
  GET_FILENAME_COMPONENT( _test ${_test} NAME_WE )

  _CTEST_FILE_CMP(${_test})

ENDFOREACH()
