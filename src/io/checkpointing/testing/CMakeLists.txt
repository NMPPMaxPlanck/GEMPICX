add_executable( test_checkpoint test_checkpoint.cpp )

if(GEMPIC_USE_CUDA) 
  setup_target_for_cuda_compilation( test_checkpoint )
endif()

target_link_libraries( test_checkpoint  AMReX::amrex AMReX::Flags_CXX  )


if("${AMReX_SPACEDIM}" STREQUAL "3")
  add_subdirectory(IOFiles_3D)
elseif("${AMReX_SPACEDIM}" STREQUAL "2")
  add_subdirectory(IOFiles_2D)
elseif("${AMReX_SPACEDIM}" STREQUAL "1")
  add_subdirectory(IOFiles_1D)
else()
  message("Dimension not implemented in AMReX")
endif()
