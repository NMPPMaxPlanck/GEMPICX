add_subdirectory( hamiltonian_splitting )

target_sources(GtestTests PRIVATE
    # test cases
    splitting_parent_class_test.cpp)