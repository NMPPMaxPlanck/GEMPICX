file(GLOB PARAMETERS "../../src/io/parameters/GEMPIC_Params.cpp")
file(GLOB FDDERHAMCOMPLEX "../../src/forms/FDDeRhamComplex/GEMPIC_FDDeRhamComplex.cpp")

target_sources(GtestTests PRIVATE
    # test cases
#    deposit_rho_test.cpp
    spline_test.cpp
    evaluate_efield_test.cpp ${PARAMETERS} ${FDDERHAMCOMPLEX})
