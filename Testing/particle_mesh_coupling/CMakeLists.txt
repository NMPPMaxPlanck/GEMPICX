# enable gtest tests
enable_testing()

add_executable(
  deposit_rho_test
  deposit_rho_test.cpp
)
target_link_libraries(
  deposit_rho_test
  AMReX::amrex AMReX::Flags_CXX
  GTest::gtest_main
  )

if (USE_CUDA)
  setup_target_for_cuda_compilation( deposit_rho_test )
endif()

include(GoogleTest)
gtest_discover_tests(deposit_rho_test)