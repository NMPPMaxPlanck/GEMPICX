file(GLOB PARAMETERS "../../src/io/parameters/GEMPIC_Params.cpp")
file(GLOB FDDERHAMCOMPLEX "../../src/forms/FDDeRhamComplex/GEMPIC_FDDeRhamComplex.cpp")
file(GLOB DERIVATIVES "../../src/forms/GEMPIC_extDerivatives.cpp")

target_sources(GtestTests PRIVATE
    FDDeRhamComplex_test.cpp ${PARAMETERS} ${FDDERHAMCOMPLEX})
