add_executable ( test_gempic_norm test_gempic_norm.cpp)
add_executable ( test_FillBoundary test_FillBoundary.cpp )
add_executable ( test_OverrideSync test_OverrideSync.cpp )
add_executable ( test_SumBoundary test_SumBoundary.cpp )
add_executable ( test_scalarprod test_scalarprod.cpp )

#message("#########")
#message( ${AMReX_DIR} )
#message( " ######")
target_link_libraries( test_gempic_norm tinyexpr gempic_poisson AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_FillBoundary gempic_poisson AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_OverrideSync gempic_poisson AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_SumBoundary tinyexpr gempic_poisson AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_scalarprod tinyexpr gempic_poisson AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )

######## Ctest

FILE(GLOB _tests *.expected_output)
LIST(SORT _tests)

FOREACH(_test ${_tests})
  GET_FILENAME_COMPONENT( _test ${_test} NAME_WE )

  _CTEST_FILE_CMP(${_test})

ENDFOREACH() 
