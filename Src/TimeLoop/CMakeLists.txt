target_sources(GEMPICX PRIVATE
  GEMPIC_RungeKutta.cpp
  GEMPIC_Splitting.cpp
  GEMPIC_DiscreteTime.cpp
  GEMPIC_TimeStepperBase.cpp
  GEMPIC_TimeStepper.cpp)

target_include_directories(GEMPICX INTERFACE
                           ${CMAKE_CURRENT_LIST_DIR})
