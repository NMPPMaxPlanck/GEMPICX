file(GLOB VERBOSITY "../../Src/Utils/GEMPIC_Verbosity.cpp")
file(GLOB PARAMETERS "../../Src/Io/Parameters/GEMPIC_Parameters.cpp")
file(GLOB FDDERHAMCOMPLEX "../../Src/Forms/FDDeRhamComplex/GEMPIC_FDDeRhamComplex.cpp")
file(GLOB DERIVATIVES "../../Src/Forms/GEMPIC_ExtDerivatives.cpp")

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