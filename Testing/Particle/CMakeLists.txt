set(TMP_TEST_SOURCES ParticleIterator_test.cpp
    )
if("${AMReX_SPACEDIM}" STREQUAL "3")
  LIST(APPEND TMP_TEST_SOURCES Sampler_test.cpp)
endif()
target_sources(GtestTests PRIVATE ${TMP_TEST_SOURCES})
