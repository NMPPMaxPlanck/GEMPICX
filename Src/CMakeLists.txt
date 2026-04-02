# Header files (all directories. Add here if you create a new one)
list(APPEND GEMPICX_HEADER_DIRECTORIES
     "Src"
     "Src/FieldSolvers"
     "Src/FieldSolvers/Poisson"
     "Src/FieldSolvers/Quasineutral"
     "Src/Filter"
     "Src/Forms"
     "Src/Forms/FDDeRhamComplex"
     "Src/HyperbolicSolvers"
     "Src/Io/Diagnostics"
     "Src/Io/Diagnostics/FullDiagnostics"
     "Src/Io/Diagnostics/FullDiagnostics/DiagnosticOutputProcessors"
     "Src/Io/Diagnostics/FullDiagnostics/FlushFormats"
     "Src/Io/Diagnostics/ReducedDiagnostics"
     "Src/Io/Parameters"
     "Src/Io/Profiling"
     "Src/Particle"
     "Src/ParticleMeshCoupling"
     "Src/TimeLoop"
     "Src/Utils")
list(TRANSFORM GEMPICX_HEADER_DIRECTORIES PREPEND "${PROJECT_SOURCE_DIR}/")

target_include_directories(GEMPICX PUBLIC
                           ${GEMPICX_HEADER_DIRECTORIES})

target_sources(GEMPICX PRIVATE
               GEMPIC_ComputationalDomain.cpp)

add_subdirectory(FieldSolvers)
add_subdirectory(Filter)
add_subdirectory(Forms)
add_subdirectory(Io)
add_subdirectory(Particle)
add_subdirectory(TimeLoop)
add_subdirectory(Utils)
