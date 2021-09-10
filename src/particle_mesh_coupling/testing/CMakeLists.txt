add_executable ( test_particle_groups test_particle_groups.cpp )
add_executable ( test_initialize test_initialize.cpp  )
add_executable ( test_splines test_splines.cpp )
add_executable ( test_AMReX_NormMasks test_AMReX_NormMasks.cpp )
add_executable ( test_deposit_rho test_deposit_rho.cpp  )
add_executable ( test_one_part test_one_part.cpp )

if (USE_CUDA) 
	setup_target_for_cuda_compilation ( test_particle_groups ) 
	setup_target_for_cuda_compilation ( test_AMReX_NormMasks )
        setup_target_for_cuda_compilation ( test_initialize      )
        setup_target_for_cuda_compilation ( test_splines         )
        setup_target_for_cuda_compilation ( test_deposit_rho     )
        setup_target_for_cuda_compilation ( test_one_part        )
endif()

#message( ${AMReX_DIR} )
#message( " ######")
target_link_libraries( test_particle_groups tinyexpr  AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran  )
target_link_libraries( test_AMReX_NormMasks tinyexpr AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran  )
target_link_libraries( test_initialize tinyexpr AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran  )
target_link_libraries( test_splines tinyexpr AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran  )
target_link_libraries( test_deposit_rho tinyexpr AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran  )
target_link_libraries( test_one_part gempic_poisson tinyexpr AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran  )


######## Ctest

FILE(GLOB _tests *.expected_output)
LIST(SORT _tests)

FOREACH(_test ${_tests})
  GET_FILENAME_COMPONENT( _test ${_test} NAME_WE )

  _CTEST_FILE_CMP(${_test})

ENDFOREACH()
