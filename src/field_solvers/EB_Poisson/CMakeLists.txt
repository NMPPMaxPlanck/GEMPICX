add_executable ( all_folder main.cpp initProbEB.cpp MAG2dEB.cpp MAG2dEBPlotfile.cpp )
#add_executable ( initProbEB initProbEB.cpp )
#add_executable ( MAG2dEB MAG2dEB.cpp )
#add_executable ( MAG2dEBPlotfile MAG2dEBPlotfile.cpp )

target_link_libraries( all_folder AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
#target_link_libraries( initProbEB AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
#target_link_libraries( MAG2dEB AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
#target_link_libraries( MAG2dEBPlotfile AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
#target_link_libraries( main AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )

include_directories("${CMAKE_SOURCE_DIR}")
include_directories("${CMAKE_SOURCE_DIR}/src/field_solvers/EB_Poisson")

######## Ctest

FILE(GLOB _tests *.expected_output)
LIST(SORT _tests)

FOREACH(_test ${_tests})
  GET_FILENAME_COMPONENT( _test ${_test} NAME_WE )

  _CTEST_FILE_CMP(${_test})

ENDFOREACH()
