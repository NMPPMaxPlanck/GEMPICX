file(GLOB VERBOSITY "../../src/utils/GEMPIC_Verbosity.cpp")
file(GLOB PARAMETERS "../../src/io/parameters/GEMPIC_Parameters.cpp")
file(GLOB FDDERHAMCOMPLEX "../../src/forms/fd_derham_complex/GEMPIC_FDDeRhamComplex.cpp")
file(GLOB DERIVATIVES "../../src/forms/GEMPIC_ExtDerivatives.cpp")

target_sources(GtestTests PRIVATE
    FDDeRhamComplex_test.cpp ${PARAMETERS} ${VERBOSITY} ${FDDERHAMCOMPLEX})
