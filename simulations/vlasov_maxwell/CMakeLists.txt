add_executable ( vlasov_maxwell vlasov_maxwell.cpp )


if (USE_CUDA)
  setup_target_for_cuda_compilation( vlasov_maxwell )
endif()

#message("#########")
#message( ${AMReX_DIR} )
#message( " ######")
target_link_libraries( vlasov_maxwell AMReX::amrex AMReX::Flags_CXX  )


######## Ctest

FILE(GLOB _tests *.expected_output)
LIST(SORT _tests)

FOREACH(_test ${_tests})
  GET_FILENAME_COMPONENT( _test ${_test} NAME_WE )

  _CTEST_FILE_CMP(${_test})

ENDFOREACH()
