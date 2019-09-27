add_executable ( test_maxwell_yee_node test_maxwell_yee_node.cpp )
add_executable ( test_maxwell_yee_cell test_maxwell_yee_cell.cpp )

#message("#########")
#message( ${AMReX_DIR} )
#message( " ######")
target_link_libraries( test_maxwell_yee_node ${AMReX_LIBRARIES} )
#target_link_libraries( test_maxwell_yee gem_fields )
target_link_libraries( test_maxwell_yee_cell ${AMReX_LIBRARIES} )

######## Ctest

FILE(GLOB _tests *.expected_output)
LIST(SORT _tests)

FOREACH(_test ${_tests})
  GET_FILENAME_COMPONENT( _test ${_test} NAME_WE )

  _CTEST_FILE_CMP(${_test})

ENDFOREACH()
