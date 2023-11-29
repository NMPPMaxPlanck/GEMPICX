file(GLOB VERBOSITY "../../../utils/GEMPIC_Verbosity.cpp")
file(GLOB PARAMETERS "../GEMPIC_parameters.cpp")
file(GLOB FDDERHAMCOMPLEX "../../../forms/FDDeRhamComplex/GEMPIC_FDDeRhamComplex.cpp")
file(GLOB DERIVATIVES "../../../forms/GEMPIC_extDerivatives.cpp")

add_executable( test_parameters test_parameters.cpp ${PARAMETERS} ${VERBOSITY} ${FDDERHAMCOMPLEX} ${DERIVATIVES} )

#add_executable( test_parser test_parser.cpp )

if(GEMPIC_USE_CUDA) 
  setup_target_for_cuda_compilation( test_parameters )
#  setup_target_for_cuda_compilation( test_parser )
endif()

target_link_libraries( test_parameters AMReX::amrex AMReX::Flags_CXX )
#target_link_libraries( test_parser AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )

list(SORT _tests)

if("${AMReX_SPACEDIM}" STREQUAL "3")
  add_subdirectory(IOFiles_3D)
elseif("${AMReX_SPACEDIM}" STREQUAL "2")
  add_subdirectory(IOFiles_2D)
elseif("${AMReX_SPACEDIM}" STREQUAL "1")
  add_subdirectory(IOFiles_1D)
else()
  message("Dimension not implemented in AMReX")
endif()



