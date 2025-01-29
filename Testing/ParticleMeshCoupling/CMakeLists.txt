set(TMP_TEST_SOURCES 
    DepositRho_test.cpp
    DepositJ_test.cpp
    DepositJcrossB_test.cpp
    DepositDelDotS_test.cpp
    EvaluateBField_test.cpp
    EvaluateEField_test.cpp
    SplineBase_test.cpp
    SplineWithPrimitive_test.cpp
    BSpline_test.cpp
    )
if("${AMReX_SPACEDIM}" STREQUAL "3") # accumulate_J_integrate_B_test is not ready
  LIST(APPEND TMP_TEST_SOURCES AccumulateJIntegrateB_test.cpp)
endif()
target_sources(GtestTests PRIVATE ${TMP_TEST_SOURCES})
