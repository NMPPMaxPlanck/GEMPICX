add_executable ( test_particle_groups test_particle_groups.cpp ${CMAKE_SOURCE_DIR}/src/utils/tinyexpr.c )
add_executable ( test_initialize test_initialize.cpp )
add_executable ( test_splines test_splines.cpp )

#message("#########")
#message( ${AMReX_DIR} )
#message( " ######")
target_link_libraries( test_particle_groups AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran  )
target_link_libraries( test_initialize AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran  )
target_link_libraries( test_splines AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran  )
include_directories("${CMAKE_SOURCE_DIR}/src/particle_pusher")
include_directories("${CMAKE_SOURCE_DIR}/src/infrastructure")
include_directories("${CMAKE_SOURCE_DIR}/src/field_solvers")
include_directories("${CMAKE_SOURCE_DIR}/src/initializer")
include_directories("${CMAKE_SOURCE_DIR}/src/diagnostics")
include_directories("${CMAKE_SOURCE_DIR}/src/utils")
include_directories("${CMAKE_SOURCE_DIR}/src/sampler")
include_directories("${CMAKE_SOURCE_DIR}")

######## Ctest

FILE(GLOB _tests *.expected_output)
LIST(SORT _tests)

FOREACH(_test ${_tests})
  GET_FILENAME_COMPONENT( _test ${_test} NAME_WE )

  _CTEST_FILE_CMP(${_test})

ENDFOREACH()
