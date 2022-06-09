######## Ctest
FILE(GLOB _tests *.expected_output)

LIST(SORT _tests)

FOREACH(_test ${_tests})
  GET_FILENAME_COMPONENT( _test ${_test} NAME_WE )

  _CTEST_FILE_CMP(${_test})


ENDFOREACH()