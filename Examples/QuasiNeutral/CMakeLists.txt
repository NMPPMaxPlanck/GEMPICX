file(GLOB VERBOSITY "${CMAKE_SOURCE_DIR}/Src/Utils/GEMPIC_Verbosity.cpp")
file(GLOB PARAMETERS "${CMAKE_SOURCE_DIR}/Src/Io/Parameters/GEMPIC_Parameters.cpp")
file(GLOB FDDERHAMCOMPLEX "${CMAKE_SOURCE_DIR}/Src/Forms/FDDeRhamComplex/GEMPIC_FDDeRhamComplex.cpp")
file(GLOB DERIVATIVES "${CMAKE_SOURCE_DIR}/Src/Forms/GEMPIC_ExtDerivatives.cpp")
file(GLOB POISSON "${CMAKE_SOURCE_DIR}/Src/FieldSolvers/Poisson/GEMPIC_PoissonSolver.cpp")

add_executable( quasi_neutral QuasiNeutralSimulation.cpp ${VERBOSITY} ${PARAMETERS} ${FDDERHAMCOMPLEX} ${DERIVATIVES})

if(GEMPIC_USE_CUDA) 
  setup_target_for_cuda_compilation( quasi_neutral ) 
endif()

target_link_libraries( quasi_neutral  AMReX::amrex AMReX::Flags_CXX  )

