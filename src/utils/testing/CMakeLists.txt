add_executable ( test_gempic_norm test_gempic_norm.cpp )
add_executable ( test_checkpoint test_checkpoint.cpp ${CMAKE_SOURCE_DIR}/src/utils/tinyexpr.c )
add_executable ( test_tinyexpr test_tinyexpr.cpp ${CMAKE_SOURCE_DIR}/src/utils/tinyexpr.c )

#message("#########")
#message( ${AMReX_DIR} )
#message( " ######")
target_link_libraries( test_gempic_norm AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_checkpoint AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( test_tinyexpr AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
include_directories("${CMAKE_SOURCE_DIR}/src/utils")
include_directories("${CMAKE_SOURCE_DIR}/src/initializer")
include_directories("${CMAKE_SOURCE_DIR}/src/infrastructure")
include_directories("${CMAKE_SOURCE_DIR}/src/field_solvers")
include_directories("${CMAKE_SOURCE_DIR}/src/particle_pusher")
include_directories("${CMAKE_SOURCE_DIR}/src/sampler")
include_directories("${CMAKE_SOURCE_DIR}")

######## Ctest

FILE(GLOB _tests *.expected_output)
LIST(SORT _tests)

FOREACH(_test ${_tests})
  GET_FILENAME_COMPONENT( _test ${_test} NAME_WE )

  _CTEST_FILE_CMP(${_test})

ENDFOREACH() 
