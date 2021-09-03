add_executable ( test_particle_groups test_particle_groups.cpp ${CMAKE_SOURCE_DIR}/third_party/tinyexpr.c ${CMAKE_SOURCE_DIR}/src/field_solvers/poisson_solvers/GEMPIC_MLNodeLaplacian_FD.cpp)
add_executable ( test_initialize test_initialize.cpp ${CMAKE_SOURCE_DIR}/third_party/tinyexpr.c ${CMAKE_SOURCE_DIR}/src/field_solvers/poisson_solvers/GEMPIC_MLNodeLaplacian_FD.cpp)
add_executable ( test_splines test_splines.cpp ${CMAKE_SOURCE_DIR}/src/field_solvers/poisson_solvers/GEMPIC_MLNodeLaplacian_FD.cpp)
add_executable ( test_norm_masks test_norm_masks.cpp ${CMAKE_SOURCE_DIR}/third_party/tinyexpr.c ${CMAKE_SOURCE_DIR}/src/field_solvers/poisson_solvers/GEMPIC_MLNodeLaplacian_FD.cpp)
add_executable ( test_deposit_rho test_deposit_rho.cpp ${CMAKE_SOURCE_DIR}/third_party/tinyexpr.c ${CMAKE_SOURCE_DIR}/src/field_solvers/poisson_solvers/GEMPIC_MLNodeLaplacian_FD.cpp)
add_executable ( test_one_part test_one_part.cpp ${CMAKE_SOURCE_DIR}/third_party/tinyexpr.c ${CMAKE_SOURCE_DIR}/src/field_solvers/poisson_solvers/GEMPIC_MLNodeLaplacian_FD.cpp)

#message("#########")
#message( ${AMReX_DIR} )
#message( " ######")
target_link_libraries( test_particle_groups AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran  )
target_link_libraries( test_norm_masks AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran  )
target_link_libraries( test_initialize AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran  )
target_link_libraries( test_splines AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran  )
target_link_libraries( test_deposit_rho AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran  )
target_link_libraries( test_one_part AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran  )


######## Ctest

FILE(GLOB _tests *.expected_output)
LIST(SORT _tests)

FOREACH(_test ${_tests})
  GET_FILENAME_COMPONENT( _test ${_test} NAME_WE )

  _CTEST_FILE_CMP(${_test})

ENDFOREACH()
