add_executable ( PIC PIC.cpp )

#message("#########")
#message( ${AMReX_DIR} )
#message( " ######")
target_link_libraries( PIC ${AMReX_LIBRARIES} )
include_directories("${CMAKE_SOURCE_DIR}/src/particle_pusher")
include_directories("${CMAKE_SOURCE_DIR}/src/infrastructure")
include_directories("${CMAKE_SOURCE_DIR}/src/field_solvers")
include_directories("${CMAKE_SOURCE_DIR}/src/sampler")
include_directories("${CMAKE_SOURCE_DIR}/src/initializer")
include_directories("${CMAKE_SOURCE_DIR}/src/time_loop")
include_directories("${CMAKE_SOURCE_DIR}/src/diagnostics")

######## Ctest

FILE(GLOB _tests *.expected_output)
LIST(SORT _tests)

FOREACH(_test ${_tests})
  GET_FILENAME_COMPONENT( _test ${_test} NAME_WE )

  _CTEST_FILE_CMP(${_test})

ENDFOREACH()
