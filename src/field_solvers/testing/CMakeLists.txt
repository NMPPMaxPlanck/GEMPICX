add_executable ( test_maxwell_yee test_maxwell_yee.cpp ${CMAKE_SOURCE_DIR}/src/utils/tinyexpr.c ${CMAKE_SOURCE_DIR}/src/field_solvers/poisson_solvers/GEMPIC_MLNodeLaplacian_FD.cpp)
add_executable ( test_maxwell_yee_order test_maxwell_yee_order.cpp ${CMAKE_SOURCE_DIR}/src/utils/tinyexpr.c ${CMAKE_SOURCE_DIR}/src/field_solvers/poisson_solvers/GEMPIC_MLNodeLaplacian_FD.cpp)
add_executable ( test_poisson_CG test_poisson_CG.cpp ${CMAKE_SOURCE_DIR}/src/utils/tinyexpr.c ${CMAKE_SOURCE_DIR}/src/field_solvers/poisson_solvers/GEMPIC_MLNodeLaplacian_FD.cpp)
add_executable ( test_ihodge_CG test_ihodge_CG.cpp ${CMAKE_SOURCE_DIR}/src/utils/tinyexpr.c ${CMAKE_SOURCE_DIR}/src/field_solvers/poisson_solvers/GEMPIC_MLNodeLaplacian_FD.cpp)
add_executable ( test_ampere_faraday test_ampere_faraday.cpp ${CMAKE_SOURCE_DIR}/src/utils/tinyexpr.c ${CMAKE_SOURCE_DIR}/src/field_solvers/poisson_solvers/GEMPIC_MLNodeLaplacian_FD.cpp)
#add_executable ( test_FillBD test_FillBD.cpp )

#message("#########")
#message( ${AMReX_DIR} )
#message( " ######")
target_link_libraries( test_maxwell_yee AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_maxwell_yee_order AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_poisson_CG AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_ihodge_CG AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_ampere_faraday AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )

#target_link_libraries( test_FillBD ${AMReX_LIBRARIES} )
include_directories("${CMAKE_SOURCE_DIR}/src/field_solvers")
include_directories("${CMAKE_SOURCE_DIR}/src/field_solvers/poisson_solvers")
include_directories("${CMAKE_SOURCE_DIR}/src/infrastructure")
include_directories("${CMAKE_SOURCE_DIR}/src/particle_pusher")
include_directories("${CMAKE_SOURCE_DIR}/src/diagnostics")
include_directories("${CMAKE_SOURCE_DIR}/src/utils")
include_directories("${CMAKE_SOURCE_DIR}/src/sampler")
include_directories("${CMAKE_SOURCE_DIR}/src/vlasov_maxwell")
include_directories("${CMAKE_SOURCE_DIR}")

######## Ctest

FILE(GLOB _tests *.expected_output)
LIST(SORT _tests)

FOREACH(_test ${_tests})
  GET_FILENAME_COMPONENT( _test ${_test} NAME_WE )

  _CTEST_FILE_CMP(${_test})

ENDFOREACH()
