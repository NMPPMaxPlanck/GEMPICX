gempic_FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/googletest-src
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable (GtestTests GEMPIC_UnitTests.cpp)

target_include_directories(GtestTests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

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
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=--disable-optimizer-constants" )
endif(GEMPIC_USE_CUDA)
  
target_link_libraries(
  GtestTests
  PRIVATE GEMPICX
  GTest::gtest_main
  GTest::gmock
  )

gtest_discover_tests(
  GtestTests
  PROPERTIES DISCOVERY_TIMEOUT 15
  EXTRA_ARGS "--gtest_color=auto")
