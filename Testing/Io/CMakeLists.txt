# add_subdirectory( hamiltonian_splitting )

target_sources(GtestTests PRIVATE
    # test cases
    Parameters_test.cpp
    FullDiagnostics_test.cpp
    ReducedDiagnostics_test.cpp
    )