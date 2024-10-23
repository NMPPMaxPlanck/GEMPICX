set(TMP_TEST_SOURCES 
    PoissonSolver_test.cpp
    )
if(AMReX_HYPRE)
  LIST(APPEND TMP_TEST_SOURCES HyprePoissonSolver_test.cpp)
endif()
target_sources(GtestTests PRIVATE ${TMP_TEST_SOURCES})
