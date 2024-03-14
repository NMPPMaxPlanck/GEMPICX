file(GLOB VERBOSITY "../../../Utils/GEMPIC_Verbosity.cpp")
file(GLOB PARAMETERS "../GEMPIC_Parameters.cpp")
file(GLOB FDDERHAMCOMPLEX "../../../Forms/FDDeRhamComplex/GEMPIC_FDDeRhamComplex.cpp")
file(GLOB DERIVATIVES "../../../Forms/GEMPIC_ExtDerivatives.cpp")

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
  add_subdirectory(IOFiles3D)
elseif("${AMReX_SPACEDIM}" STREQUAL "2")
  add_subdirectory(IOFiles2D)
elseif("${AMReX_SPACEDIM}" STREQUAL "1")
  add_subdirectory(IOFiles1D)
else()
  message("Dimension not implemented in AMReX")
endif()



