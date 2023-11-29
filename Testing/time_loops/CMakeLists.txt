file(GLOB VERBOSITY "../../src/utils/GEMPIC_Verbosity.cpp")
file(GLOB PARAMETERS "../../src/io/parameters/GEMPIC_parameters.cpp")
file(GLOB FDDERHAMCOMPLEX "../../src/forms/FDDeRhamComplex/GEMPIC_FDDeRhamComplex.cpp")
file(GLOB DERIVATIVES "../../src/forms/GEMPIC_extDerivatives.cpp")

# add_subdirectory( hamiltonian_splitting )

target_sources(GtestTests PRIVATE
    # test cases
    operatorHamilton_test.cpp ${PARAMETERS} ${VERBOSITY} ${FDDERHAMCOMPLEX} ${DERIVATIVES}
    )#    hs_zigzag_test.cpp ${PARAMETERS} ${VERBOSITY} ${FDDERHAMCOMPLEX} ${DERIVATIVES})