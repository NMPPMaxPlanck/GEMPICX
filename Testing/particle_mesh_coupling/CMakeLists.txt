file(GLOB PARAMETERS "../../src/io/parameters/GEMPIC_Params.cpp")
file(GLOB FDDERHAMCOMPLEX "../../src/forms/FDDeRhamComplex/GEMPIC_FDDeRhamComplex.cpp")

# add_executable ( evaluate_efield_test ${PARAMETERS} ${FDDERHAMCOMPLEX} evaluate_efield_test.cpp)

target_sources(GtestTests PRIVATE
    # test cases
    deposit_rho_test.cpp
    evaluate_efield_test.cpp ${PARAMETERS} ${FDDERHAMCOMPLEX})
