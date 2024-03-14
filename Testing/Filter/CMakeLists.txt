
file(GLOB VERBOSITY "../../Src/Utils/GEMPIC_Verbosity.cpp")
file(GLOB PARAMETERS "../../Src/Io/Parameters/GEMPIC_Parameters.cpp")
file(GLOB FDDERHAMCOMPLEX "../../Src/Forms/FDDeRhamComplex/GEMPIC_FDDeRhamComplex.cpp")

target_sources(GtestTests PRIVATE
    # test cases
    Filter_test.cpp ${PARAMETERS} ${VERBOSITY} ${FDDERHAMCOMPLEX}
    )