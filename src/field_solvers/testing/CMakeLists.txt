add_executable ( test_maxwell_yee test_maxwell_yee.cpp )
add_executable ( test_maxwell_yee_order test_maxwell_yee_order.cpp )
add_executable ( test_poisson_CG test_poisson_CG.cpp )
add_executable ( test_ihodge_CG test_ihodge_CG.cpp )
add_executable ( test_ampere_faraday test_ampere_faraday.cpp )
add_executable ( test_AMReX_atomic_add test_AMReX_atomic_add.cpp )
#add_executable ( test_FillBD test_FillBD.cpp )

#message("#########")
#message( ${AMReX_DIR} )
#message( " ######")
target_link_libraries( test_maxwell_yee tinyexpr gempic_poisson AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_maxwell_yee_order tinyexpr gempic_poisson AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_poisson_CG tinyexpr gempic_poisson AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_ihodge_CG tinyexpr gempic_poisson AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_ampere_faraday tinyexpr gempic_poisson AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_AMReX_atomic_add tinyexpr gempic_poisson AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )

#target_link_libraries( test_FillBD ${AMReX_LIBRARIES} )


######## Ctest

FILE(GLOB _tests *.expected_output)
LIST(SORT _tests)

FOREACH(_test ${_tests})
  GET_FILENAME_COMPONENT( _test ${_test} NAME_WE )

  _CTEST_FILE_CMP(${_test})

ENDFOREACH()
