add_executable(maxwell Maxwell.cpp GEMPIC_MaxwellInit.cpp)

target_link_libraries(maxwell GEMPICX::GEMPICX)
