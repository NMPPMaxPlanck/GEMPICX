file(GLOB VERBOSITY "../../src/utils/GEMPIC_Verbosity.cpp")
file(GLOB PARAMETERS "../../src/io/parameters/GEMPIC_parameters.cpp")
file(GLOB FDDERHAMCOMPLEX "../../src/forms/FDDeRhamComplex/GEMPIC_FDDeRhamComplex.cpp")
file(GLOB DERIVATIVES "../../src/forms/GEMPIC_extDerivatives.cpp")

if("${AMReX_SPACEDIM}" STREQUAL "3") # accumulate_J_integrate_B_test is not ready
target_sources(GtestTests PRIVATE
    # test cases
    accumulate_J_integrate_B_test.cpp
    deposit_rho_test.cpp
    evaluate_bfield_test.cpp
    evaluate_efield_test.cpp ${PARAMETERS} ${VERBOSITY} ${FDDERHAMCOMPLEX}
    SplineBase_test.cpp
    SplineWithPrimitive_test.cpp)
else()
target_sources(GtestTests PRIVATE
    # test cases
    deposit_rho_test.cpp
    evaluate_bfield_test.cpp
    evaluate_efield_test.cpp ${PARAMETERS} ${VERBOSITY} ${FDDERHAMCOMPLEX}
    SplineBase_test.cpp
    SplineWithPrimitive_test.cpp)
endif()