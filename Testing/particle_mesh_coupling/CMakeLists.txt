file(GLOB PARAMETERS "../../src/io/parameters/GEMPIC_Params.cpp")
file(GLOB FDDERHAMCOMPLEX "../../src/forms/FDDeRhamComplex/GEMPIC_FDDeRhamComplex.cpp")

target_sources(GtestTests PRIVATE
    # test cases
    # deposit_rho_test.cpp)
    # spline_test.cpp)
    # evaluate_efield_test.cpp ${PARAMETERS} ${FDDERHAMCOMPLEX})
    evaluate_bfield_test.cpp ${PARAMETERS} ${FDDERHAMCOMPLEX})
    # push_v_efield_test.cpp)
    # accumulate_j_update_v_C2_test.cpp ${PARAMETERS} ${FDDERHAMCOMPLEX})
