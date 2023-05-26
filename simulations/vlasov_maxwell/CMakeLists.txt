add_executable( vlasov_maxwell vlasov_maxwell.cpp )


if(GEMPIC_USE_CUDA)
  setup_target_for_cuda_compilation( vlasov_maxwell )
endif()

target_link_libraries( vlasov_maxwell AMReX::amrex AMReX::Flags_CXX  )


######## Ctest

file(GLOB _tests *.expected_output)
list(SORT _tests)

foreach(_test ${_tests})
  get_filename_component( _test ${_test} NAME_WE )

  _CTEST_FILE_CMP(${_test})

endforeach()
