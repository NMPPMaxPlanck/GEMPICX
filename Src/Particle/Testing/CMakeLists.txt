file(GLOB VERBOSITY "../../Utils/GEMPIC_Verbosity.cpp")
file(GLOB PARAMETERS "../../Io/Parameters/GEMPIC_Parameters.cpp")
add_executable( test_sampler ${PARAMETERS} ${VERBOSITY} test_sampler.cpp )

if(GEMPIC_USE_CUDA) 
  setup_target_for_cuda_compilation( test_sampler ) 
endif()

target_link_libraries( test_sampler  AMReX::amrex AMReX::Flags_CXX  )

if("${AMReX_SPACEDIM}" STREQUAL "3")
  add_subdirectory(IOFiles3D)
elseif("${AMReX_SPACEDIM}" STREQUAL "2")
  add_subdirectory(IOFiles2D)
elseif("${AMReX_SPACEDIM}" STREQUAL "1")
  add_subdirectory(IOFiles1D)
else()
  message("Dimension not implemented in AMReX")
endif()
