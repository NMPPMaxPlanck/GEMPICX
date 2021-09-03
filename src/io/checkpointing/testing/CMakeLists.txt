add_executable ( test_vlasov_maxwell_by test_vlasov_maxwell_by.cpp )
add_executable ( test_vlasov_maxwell_by_lower test_vlasov_maxwell_by_lower.cpp )
add_executable ( test_vlasov_maxwell_hs test_vlasov_maxwell_hs.cpp  )
add_executable ( test_vlasov_maxwell_hs_multispecies test_vlasov_maxwell_hs_multispecies.cpp  )
add_executable ( test_vlasov_maxwell_hs_lower test_vlasov_maxwell_hs_lower.cpp )
add_executable ( test_vlasov_maxwell_hs_all test_vlasov_maxwell_hs_all.cpp )
add_executable ( test_vlasov_maxwell_hs_all_lower test_vlasov_maxwell_hs_all_lower.cpp )
add_executable ( test_vlasov_maxwell_hs_zigzag_C2 test_vlasov_maxwell_hs_zigzag_C2.cpp )
add_executable ( test_checkpoint test_checkpoint.cpp )


#message("#########")
#message( ${AMReX_DIR} )
#message( " ######")
target_link_libraries( test_vlasov_maxwell_by gempic_poisson tinyexpr AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_vlasov_maxwell_by_lower gempic_poisson tinyexpr AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_vlasov_maxwell_hs gempic_poisson tinyexpr AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_vlasov_maxwell_hs_multispecies gempic_poisson tinyexpr AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_vlasov_maxwell_hs_lower gempic_poisson tinyexpr AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_vlasov_maxwell_hs_all gempic_poisson tinyexpr AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_vlasov_maxwell_hs_all_lower gempic_poisson tinyexpr AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_vlasov_maxwell_hs_zigzag_C2 gempic_poisson tinyexpr AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_checkpoint gempic_poisson tinyexpr AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )


######## Ctest

FILE(GLOB _tests *.expected_output)
LIST(SORT _tests)

FOREACH(_test ${_tests})
  GET_FILENAME_COMPONENT( _test ${_test} NAME_WE )

  _CTEST_FILE_CMP(${_test})

ENDFOREACH()
