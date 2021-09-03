add_executable ( vlasov_maxwell vlasov_maxwell.cpp ${CMAKE_SOURCE_DIR}/src/utils/tinyexpr.c ${CMAKE_SOURCE_DIR}/src/field_solvers/poisson_solvers/GEMPIC_MLNodeLaplacian_FD.cpp)

#message("#########")
#message( ${AMReX_DIR} )
#message( " ######")
target_link_libraries( vlasov_maxwell AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )


######## Ctest

FILE(GLOB _tests *.expected_output)
LIST(SORT _tests)

FOREACH(_test ${_tests})
  GET_FILENAME_COMPONENT( _test ${_test} NAME_WE )

  _CTEST_FILE_CMP(${_test})

ENDFOREACH()
