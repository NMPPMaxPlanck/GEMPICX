file(GLOB VERBOSITY "../../Src/Utils/GEMPIC_Verbosity.cpp")
file(GLOB PARAMETERS "../../Src/Io/Parameters/GEMPIC_Parameters.cpp")
file(GLOB FDDERHAMCOMPLEX "../../Src/Forms/FDDeRhamComplex/GEMPIC_FDDeRhamComplex.cpp")
file(GLOB DERIVATIVES "../../Src/Forms/GEMPIC_ExtDerivatives.cpp")

target_sources(GtestTests PRIVATE
    FDDeRhamComplex_test.cpp ${PARAMETERS} ${VERBOSITY} ${FDDERHAMCOMPLEX}
    FDDeRhamComplexProjection_test.cpp
    )
