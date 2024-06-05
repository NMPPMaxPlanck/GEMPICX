# enable gtest tests
enable_testing()

add_executable (GtestTests GEMPIC_UnitTests.cpp)

target_include_directories(GtestTests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(TestUtils)
add_subdirectory(FieldSolvers)
add_subdirectory(Filter)
add_subdirectory(Forms)
add_subdirectory(Io)
add_subdirectory(ParticleMeshCoupling)
add_subdirectory(TimeLoop)

add_custom_command(
  TARGET GtestTests POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
          ${CMAKE_CURRENT_BINARY_DIR}/InputFiles
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_CURRENT_SOURCE_DIR}/InputFiles/*
          ${CMAKE_CURRENT_BINARY_DIR}/InputFiles/
)

include(GoogleTest)
# Setup target to be compiled with CUDA and be linked against CUDA-enabled AMReX
# MUST be done AFTER all sources have been assigned to target
if(GEMPIC_USE_CUDA)
  setup_target_for_cuda_compilation(GtestTests)
endif(GEMPIC_USE_CUDA)
  
target_link_libraries(
  GtestTests
  PRIVATE GEMPICX
  GTest::gtest_main
  GTest::gmock
  )

gtest_discover_tests(
  GtestTests
  EXTRA_ARGS "--gtest_color=auto")