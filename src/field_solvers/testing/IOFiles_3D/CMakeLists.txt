######## Ctest
file(GLOB _tests *.expected_output)

list(SORT _tests)

foreach(_test ${_tests})
  get_filename_component( _test ${_test} NAME_WE )

  _CTEST_FILE_CMP(${_test})


endforeach()