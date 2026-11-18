target_sources(GEMPICX PRIVATE
  GEMPIC_Version.cpp
  GEMPIC_NumericalIntegrationDifferentiation.cpp
  GEMPIC_NumTools.cpp
  GEMPIC_Verbosity.cpp)

gempic_create_and_link_cmake_constants_file(GEMPIC_CMakeVersionConstants.H.in)