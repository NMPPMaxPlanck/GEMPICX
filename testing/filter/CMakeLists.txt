
file(GLOB VERBOSITY "../../src/utils/GEMPIC_Verbosity.cpp")
file(GLOB PARAMETERS "../../src/io/parameters/GEMPIC_Parameters.cpp")
file(GLOB FDDERHAMCOMPLEX "../../src/forms/fd_derham_complex/GEMPIC_FDDeRhamComplex.cpp")

target_sources(GtestTests PRIVATE
    # test cases
    Filter_test.cpp ${PARAMETERS} ${VERBOSITY} ${FDDERHAMCOMPLEX}
    )