file(GLOB VERBOSITY "../../Src/Utils/GEMPIC_Verbosity.cpp")
file(GLOB PARAMETERS "../../Src/Io/Parameters/GEMPIC_Parameters.cpp")
file(GLOB FDDERHAMCOMPLEX "../../Src/Forms/FDDeRhamComplex/GEMPIC_FDDeRhamComplex.cpp")
file(GLOB DERIVATIVES "../../Src/Forms/GEMPIC_ExtDerivatives.cpp")

# add_subdirectory( hamiltonian_splitting )

target_sources(GtestTests PRIVATE
    # test cases
    Parameters_test.cpp ${PARAMETERS} ${VERBOSITY} ${FDDERHAMCOMPLEX} ${DERIVATIVES}
    FullDiagnostics_test.cpp
    ReducedDiagnostics_test.cpp
    )