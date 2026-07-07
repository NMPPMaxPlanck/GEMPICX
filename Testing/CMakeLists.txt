include(${PROJECT_SOURCE_DIR}/cmake/gempic_GoogleTestConfig.cmake)

add_executable(GtestTests GEMPIC_UnitTests.cpp)
gempic_set_compile_options(GtestTests)

target_include_directories(GtestTests PRIVATE ${CMAKE_CURRENT_LIST_DIR})

add_subdirectory(AMR)
add_subdirectory(FieldSolvers)
add_subdirectory(Filter)
add_subdirectory(Forms)
add_subdirectory(Io)
add_subdirectory(Particle)
add_subdirectory(ParticleMeshCoupling)
add_subdirectory(Utils)
add_subdirectory(TestUtils)
add_subdirectory(TimeLoop)
if(GEMPIC_BUILD_EXAMPLES)
  add_subdirectory(Examples)
endif()

target_sources(GtestTests PRIVATE
                ComputationalDomain_test.cpp)

add_custom_command(
  TARGET GtestTests POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
          ${CMAKE_CURRENT_BINARY_DIR}/InputFiles
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_CURRENT_SOURCE_DIR}/InputFiles
          ${CMAKE_CURRENT_BINARY_DIR}/InputFiles
)

include(GoogleTest)
# Setup target to be compiled with CUDA and be linked against CUDA-enabled AMReX
# MUST be done AFTER all sources have been assigned to target
if(GEMPIC_USE_CUDA)
  setup_target_for_cuda_compilation(GtestTests)
  # Too many amrex::ParserExecutor's inside ParallelFor loops. Only a problem because we gather all
  # tests in one executable.
  target_compile_options(GtestTests PRIVATE --ptxas-options=--disable-optimizer-constants)
endif()

target_link_libraries(
  GtestTests
  PRIVATE
  GEMPICX::GEMPICX
  GTest::gtest_main
  GTest::gmock
  )
if(GEMPIC_USE_HDF5)
  target_link_libraries(GtestTests PRIVATE GEMPICX::HDF5)
endif()

gtest_discover_tests(
  GtestTests
  PROPERTIES DISCOVERY_TIMEOUT 15
  EXTRA_ARGS "--gtest_color=auto")
