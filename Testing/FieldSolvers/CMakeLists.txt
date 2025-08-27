set(TMP_TEST_SOURCES 
    PoissonSolver_test.cpp
    MaxwellFDSolver_test.cpp
    )
if(AMReX_HYPRE)
  LIST(APPEND TMP_TEST_SOURCES
       HypreNonperiodicPoissonSolver_test.cpp
       HyprePoissonSolver_test.cpp
       HypreQuasineutralCurlCurl_test.cpp
       HypreQuasineutralChargeMatrix_test.cpp
       HypreQuasineutralCurlCurlplusChargeMatrix_test.cpp
       HypreQuasineutralJcrossCurl_test.cpp
       HypreQuasineutralFullQNMatrix_test.cpp
       HypreQuasineutralAPoisson_test.cpp
       HypreQuasineutralDivGradPhiMatrix_test.cpp
       HypreQuasineutralSobolDivJCurlB_test.cpp)
endif()
target_sources(GtestTests PRIVATE ${TMP_TEST_SOURCES})
