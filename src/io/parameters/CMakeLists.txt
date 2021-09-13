ADD_LIBRARY(gempic_io STATIC 
	GEMPIC_parameter_keys.cpp
	GEMPIC_parameter_i_init.cpp
	GEMPIC_parameter.cpp
        GEMPIC_parameter_config.cpp
        GEMPIC_parameter_struct.cpp)
if (USE_CUDA)
	setup_target_for_cuda_compilation ( gempic_io )
endif()
target_link_libraries( gempic_io AMReX::amrex AMReX::Flags_CXX)
