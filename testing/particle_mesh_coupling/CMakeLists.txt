file(GLOB VERBOSITY "../../src/utils/GEMPIC_Verbosity.cpp")
file(GLOB PARAMETERS "../../src/io/parameters/GEMPIC_Parameters.cpp")
file(GLOB FDDERHAMCOMPLEX "../../src/forms/fd_derham_complex/GEMPIC_FDDeRhamComplex.cpp")
file(GLOB DERIVATIVES "../../src/forms/GEMPIC_ExtDerivatives.cpp")

if("${AMReX_SPACEDIM}" STREQUAL "3") # accumulate_J_integrate_B_test is not ready
target_sources(GtestTests PRIVATE
    # test cases
    AccumulateJIntegrateB_test.cpp
    DepositRho_test.cpp
    EvaluateBField_test.cpp
    EvaluateEField_test.cpp ${PARAMETERS} ${VERBOSITY} ${FDDERHAMCOMPLEX}
    SplineBase_test.cpp
    SplineWithPrimitive_test.cpp)
else()
target_sources(GtestTests PRIVATE
    # test cases
    DepositRho_test.cpp
    EvaluateBField_test.cpp
    EvaluateEField_test.cpp ${PARAMETERS} ${VERBOSITY} ${FDDERHAMCOMPLEX}
    SplineBase_test.cpp
    SplineWithPrimitive_test.cpp)
endif()