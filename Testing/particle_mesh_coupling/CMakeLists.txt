file(GLOB PARAMETERS "../../src/io/parameters/GEMPIC_Params.cpp")
file(GLOB FDDERHAMCOMPLEX "../../src/forms/FDDeRhamComplex/GEMPIC_FDDeRhamComplex.cpp")
file(GLOB DERIVATIVES "../../src/forms/GEMPIC_extDerivatives.cpp")

target_sources(GtestTests PRIVATE
    # test cases
    accumulate_J_integrate_B_test.cpp
    deposit_rho_test.cpp
    evaluate_bfield_test.cpp
    evaluate_efield_test.cpp ${PARAMETERS} ${FDDERHAMCOMPLEX}
    SplineBase_test.cpp
    SplineWithPrimitive_test.cpp)
