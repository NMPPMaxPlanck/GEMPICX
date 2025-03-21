add_subdirectory(Io)
add_subdirectory(FieldSolvers)
add_subdirectory(Filter)
add_subdirectory(Forms)
add_subdirectory(Particle)
add_subdirectory(TimeLoop)
add_subdirectory(Utils)

target_sources(GEMPICX PRIVATE
                GEMPIC_ComputationalDomain.cpp)