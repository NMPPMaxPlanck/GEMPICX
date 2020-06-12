add_executable ( test_parse test_parse.cpp )

#message("#########")
#message( ${AMReX_DIR} )
#message( " ######")
target_link_libraries( test_parse AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
#target_link_libraries( test_FillBD ${AMReX_LIBRARIES} )
#target_link_libraries( test_heat ${AMReX_LIBRARIES} )
include_directories("${CMAKE_SOURCE_DIR}/src/Parse")

######## Ctest

FILE(GLOB _tests *.expected_output)
LIST(SORT _tests)

FOREACH(_test ${_tests})
  GET_FILENAME_COMPONENT( _test ${_test} NAME_WE )

  _CTEST_FILE_CMP(${_test})

ENDFOREACH()
