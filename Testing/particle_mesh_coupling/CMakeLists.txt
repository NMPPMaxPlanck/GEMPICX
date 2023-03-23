# enable gtest tests
enable_testing()

add_executable(
  deposit_rho_test
  deposit_rho_test.cc
)
target_link_libraries(
  deposit_rho_test
  AMReX::amrex AMReX::Flags_CXX
  GTest::gtest_main
  )

include(GoogleTest)
gtest_discover_tests(deposit_rho_test)