# file(GLOB PARAMETERS "../../src/io/parameters/GEMPIC_Params.cpp")
# file(GLOB FDDERHAMCOMPLEX "../../src/forms/FDDeRhamComplex/GEMPIC_FDDeRhamComplex.cpp")
# file(GLOB DERIVATIVES "../../src/forms/GEMPIC_extDerivatives.cpp")

target_sources(GtestTests PRIVATE
    # test cases
    # accumulate_j_update_v_C2_test.cpp)
    # deposit_rho_test.cpp)
    deposit_rho_new_spline_test.cpp)
    # evaluate_bfield_test.cpp
    # evaluate_efield_test.cpp ${PARAMETERS} ${FDDERHAMCOMPLEX}
    # e_field_scaling_test.cpp ${PARAMETERS} ${FDDERHAMCOMPLEX} ${DERIVATIVES}
    # push_v_efield_test.cpp
    # SplineBase_test.cpp
    # SplineWithPrimitive_test.cpp)
    # spline_test.cpp)
