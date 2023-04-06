# enable gtest tests
enable_testing()

add_executable ( GtestTests utest_main.cpp )

target_include_directories(GtestTests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory( test_utils )
add_subdirectory( particle_mesh_coupling )

include(GoogleTest)
# Setup target to be compiled with CUDA and be linked against CUDA-enabled AMReX
# MUST be done AFTER all sources have been assigned to target
if (USE_CUDA)
  setup_target_for_cuda_compilation( GtestTests )
endif(USE_CUDA)
  
target_link_libraries(
  GtestTests
  PUBLIC AMReX::amrex AMReX::Flags_CXX
  GTest::gtest_main
  )

gtest_discover_tests(
  GtestTests
  EXTRA_ARGS "--gtest_color=auto")