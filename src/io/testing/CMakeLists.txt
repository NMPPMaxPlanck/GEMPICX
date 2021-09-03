add_executable ( test_vlasov_maxwell_by test_vlasov_maxwell_by.cpp  ${CMAKE_SOURCE_DIR}/src/field_solvers/poisson_solvers/GEMPIC_MLNodeLaplacian_FD.cpp)
add_executable ( test_vlasov_maxwell_by_lower test_vlasov_maxwell_by_lower.cpp  ${CMAKE_SOURCE_DIR}/src/field_solvers/poisson_solvers/GEMPIC_MLNodeLaplacian_FD.cpp)
add_executable ( test_vlasov_maxwell_hs test_vlasov_maxwell_hs.cpp  ${CMAKE_SOURCE_DIR}/src/field_solvers/poisson_solvers/GEMPIC_MLNodeLaplacian_FD.cpp)
add_executable ( test_vlasov_maxwell_hs_multispecies test_vlasov_maxwell_hs_multispecies.cpp  ${CMAKE_SOURCE_DIR}/src/field_solvers/poisson_solvers/GEMPIC_MLNodeLaplacian_FD.cpp)
add_executable ( test_vlasov_maxwell_hs_lower test_vlasov_maxwell_hs_lower.cpp  ${CMAKE_SOURCE_DIR}/src/field_solvers/poisson_solvers/GEMPIC_MLNodeLaplacian_FD.cpp)
add_executable ( test_vlasov_maxwell_hs_all test_vlasov_maxwell_hs_all.cpp  ${CMAKE_SOURCE_DIR}/src/field_solvers/poisson_solvers/GEMPIC_MLNodeLaplacian_FD.cpp)
add_executable ( test_vlasov_maxwell_hs_all_lower test_vlasov_maxwell_hs_all_lower.cpp  ${CMAKE_SOURCE_DIR}/src/field_solvers/poisson_solvers/GEMPIC_MLNodeLaplacian_FD.cpp)
add_executable ( test_vlasov_maxwell_hs_zigzag_C2 test_vlasov_maxwell_hs_zigzag_C2.cpp  ${CMAKE_SOURCE_DIR}/src/field_solvers/poisson_solvers/GEMPIC_MLNodeLaplacian_FD.cpp)

#message("#########")
#message( ${AMReX_DIR} )
#message( " ######")
target_link_libraries( test_vlasov_maxwell_by tinyexpr AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_vlasov_maxwell_by_lower tinyexpr AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_vlasov_maxwell_hs tinyexpr AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_vlasov_maxwell_hs_multispecies tinyexpr AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_vlasov_maxwell_hs_lower tinyexpr AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_vlasov_maxwell_hs_all tinyexpr AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_vlasov_maxwell_hs_all_lower tinyexpr AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_vlasov_maxwell_hs_zigzag_C2 tinyexpr AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )


######## Ctest

FILE(GLOB _tests *.expected_output)
LIST(SORT _tests)

FOREACH(_test ${_tests})
  GET_FILENAME_COMPONENT( _test ${_test} NAME_WE )

  _CTEST_FILE_CMP(${_test})

ENDFOREACH()
