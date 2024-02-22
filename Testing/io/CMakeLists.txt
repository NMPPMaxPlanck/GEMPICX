file(GLOB VERBOSITY "../../src/utils/GEMPIC_Verbosity.cpp")
file(GLOB PARAMETERS "../../src/io/parameters/GEMPIC_parameters.cpp")
file(GLOB FDDERHAMCOMPLEX "../../src/forms/FDDeRhamComplex/GEMPIC_FDDeRhamComplex.cpp")
file(GLOB DERIVATIVES "../../src/forms/GEMPIC_extDerivatives.cpp")

# add_subdirectory( hamiltonian_splitting )

target_sources(GtestTests PRIVATE
    # test cases
    parameters_test.cpp ${PARAMETERS} ${VERBOSITY} ${FDDERHAMCOMPLEX} ${DERIVATIVES}
    FullDiagnostics_test.cpp
    reducedDiagnostics_test.cpp
    )