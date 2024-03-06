# enable gtest tests
enable_testing()

add_executable ( GtestTests GEMPIC_UnitTests.cpp )

target_include_directories(GtestTests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory( test_utils )
add_subdirectory( filter )
add_subdirectory( forms )
add_subdirectory( io )
add_subdirectory( particle_mesh_coupling )
add_subdirectory( time_loops )

add_custom_command(
  TARGET GtestTests POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
          ${CMAKE_CURRENT_BINARY_DIR}/input_files
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_CURRENT_SOURCE_DIR}/input_files/*
          ${CMAKE_CURRENT_BINARY_DIR}/input_files/
)

include(GoogleTest)
# Setup target to be compiled with CUDA and be linked against CUDA-enabled AMReX
# MUST be done AFTER all sources have been assigned to target
if (GEMPIC_USE_CUDA)
  setup_target_for_cuda_compilation( GtestTests )
endif(GEMPIC_USE_CUDA)
  
target_link_libraries(
  GtestTests
  PUBLIC AMReX::amrex AMReX::Flags_CXX
  GTest::gtest_main
  GTest::gmock
  )

gtest_discover_tests(
  GtestTests
  EXTRA_ARGS "--gtest_color=auto")