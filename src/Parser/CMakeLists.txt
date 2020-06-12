add_executable ( Parser GpuParser.cpp WarpXParser.cpp WarpXUtil.cpp wp_parser_c.c wp_parser_y.c wp_parser.lex.c wp_parser.tab.c)
#add_executable ( GpuParser GpuParser.cpp )
#add_executable ( WarpXParser WarpXParser.cpp )
#add_executable ( WarpXUtil WarpXUtil.cpp )
#add_executable ( wp_parser_c wp_parser_c.c )
#add_executable ( wp_parser_y wp_parser_y.c )
#add_executable ( wp_parser_lex wp_parser.lex.c )
#add_executable ( wp_parser_tab wp_parser.tab.c )


include_directories("${CMAKE_SOURCE_DIR}/src/Parser")

target_link_libraries( Parser AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
#target_link_libraries( GpuParser AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
#target_link_libraries( WarpXParser AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
#target_link_libraries( WarpXUtil AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
#target_link_libraries( wp_parser_c AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
#target_link_libraries( wp_parser_y AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
#target_link_libraries( wp_parser_lex AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )
#target_link_libraries( wp_parser_tab AMReX::amrex AMReX::Flags_CXX AMReX::Flags_Fortran )

add_subdirectory(testing)


