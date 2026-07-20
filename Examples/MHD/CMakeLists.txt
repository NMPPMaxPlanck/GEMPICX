add_executable(MHD MHD.cpp)

if(MPIDEBUG)
    target_compile_definitions(MHD PRIVATE MPIDEBUG)
endif()

target_link_libraries(MHD GEMPICX::GEMPICX)
