# enable gtest tests
enable_testing()

add_executable ( GtestTests utest_main.cpp )

target_include_directories(GtestTests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

#message("#########")
#message( ${AMReX_DIR} )
#message( " ######")
target_link_libraries(
  GtestTests
  AMReX::amrex AMReX::Flags_CXX
  GTest::gtest_main
  )
  
if (USE_CUDA)
  setup_target_for_cuda_compilation( GtestTests )
endif(USE_CUDA)

add_subdirectory( test_utils )
add_subdirectory( particle_mesh_coupling )
  
include(GoogleTest)
gtest_discover_tests(GtestTests)