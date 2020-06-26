add_subdirectory( reference_results )

add_executable ( PIC PIC.cpp ${CMAKE_SOURCE_DIR}/src/utils/tinyexpr.c )
add_executable ( AmrexSolver AmrexSolver.cpp )

#message("#########")
#message( ${AMReX_DIR} )
#message( " ######")
target_link_libraries( PIC AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
target_link_libraries( AmrexSolver AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
include_directories("${CMAKE_SOURCE_DIR}/src/particle_pusher")
include_directories("${CMAKE_SOURCE_DIR}/src/infrastructure")
include_directories("${CMAKE_SOURCE_DIR}/src/field_solvers")
include_directories("${CMAKE_SOURCE_DIR}/src/sampler")
include_directories("${CMAKE_SOURCE_DIR}/src/initializer")
include_directories("${CMAKE_SOURCE_DIR}/src/time_loop")
include_directories("${CMAKE_SOURCE_DIR}/src/diagnostics")
include_directories("${CMAKE_SOURCE_DIR}/src/utils")
include_directories("${CMAKE_SOURCE_DIR}")


######## Ctest

FILE(GLOB _tests *.expected_output)
LIST(SORT _tests)

FOREACH(_test ${_tests})
  GET_FILENAME_COMPONENT( _test ${_test} NAME_WE )

  _CTEST_FILE_CMP(${_test})

ENDFOREACH()
