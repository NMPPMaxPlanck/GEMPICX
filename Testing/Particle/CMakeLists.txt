set(TMP_TEST_SOURCES
  ParticleIterator_test.cpp
  ParticleIO_test.cpp
    )
if("${AMReX_SPACEDIM}" STREQUAL "3")
  LIST(APPEND TMP_TEST_SOURCES Sampler_test.cpp)
endif()
target_sources(GtestTests PRIVATE ${TMP_TEST_SOURCES})

add_executable(
  ParticleAdd_test
  ParticleAdd_test.cpp
  ${PROJECT_SOURCE_DIR}/Testing/TestUtils/GEMPIC_TestUtils.cpp
  )

if(GEMPIC_USE_CUDA)
  setup_target_for_cuda_compilation(ParticleAdd_test)
endif()
target_link_libraries(ParticleAdd_test
                      GEMPICX::GEMPICX
                      GTest::gtest_main)
target_include_directories(ParticleAdd_test PRIVATE ${PROJECT_SOURCE_DIR}/Testing)

add_test(NAME ParticlePlotFile_test
  COMMAND ${CMAKE_COMMAND}
    -DTEST_EXECUTABLE=$<TARGET_FILE:ParticleAdd_test>
    -DPYTHON_TEST=${CMAKE_CURRENT_SOURCE_DIR}/ReadParticles.py
  -P ${CMAKE_CURRENT_SOURCE_DIR}/ParticleAdd_test.cmake
  )

# add label to alow running independently from Google tests
set_tests_properties(
  ParticlePlotFile_test
  PROPERTIES LABELS "non-Google")
