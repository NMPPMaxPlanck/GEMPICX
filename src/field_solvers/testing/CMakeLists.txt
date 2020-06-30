add_executable ( test_maxwell_yee test_maxwell_yee.cpp ${CMAKE_SOURCE_DIR}/src/utils/tinyexpr.c)
#add_executable ( test_FillBD test_FillBD.cpp )
#add_executable ( test_heat test_heat.cpp)

#message("#########")
#message( ${AMReX_DIR} )
#message( " ######")
target_link_libraries( test_maxwell_yee AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
#target_link_libraries( test_FillBD ${AMReX_LIBRARIES} )
#target_link_libraries( test_heat ${AMReX_LIBRARIES} )
include_directories("${CMAKE_SOURCE_DIR}/src/field_solvers")
include_directories("${CMAKE_SOURCE_DIR}/src/infrastructure")
include_directories("${CMAKE_SOURCE_DIR}/src/initializer")
include_directories("${CMAKE_SOURCE_DIR}/src/utils")
include_directories("${CMAKE_SOURCE_DIR}")

######## Ctest

FILE(GLOB _tests *.expected_output)
LIST(SORT _tests)

FOREACH(_test ${_tests})
  GET_FILENAME_COMPONENT( _test ${_test} NAME_WE )

  _CTEST_FILE_CMP(${_test})

ENDFOREACH()
