######## Empty File
FILE(GLOB _tests *.expected_output)
message("Selecting 3D IO Directory")
FOREACH(_test ${_tests})
  GET_FILENAME_COMPONENT( _test ${_test} NAME_WE )

  _CTEST_FILE_CMP(${_test})

ENDFOREACH()