target_sources(GtestTests PRIVATE
    RKButcherTableau_test.cpp
    RungeKutta_test.cpp
    SimulationTime_test.cpp
    SplittingColdPlasma_test.cpp
    TimeLoop_test.cpp
    TimeLoopOrder_test.cpp
    )

add_executable(
  TimeStepperBase_test
  TimeStepperBase_test.cpp)

target_link_libraries(
  TimeStepperBase_test
  GEMPICX)

add_executable(
  TimeStepper_test
  TimeStepper_test.cpp)

target_link_libraries(
  TimeStepper_test
  GEMPICX)

if(GEMPIC_USE_CUDA)
  setup_target_for_cuda_compilation(TimeStepper_test)
endif()

add_test(
  NAME TimeStepperBase_test
  COMMAND TimeStepperBase_test TSBTest00)

add_test(
  NAME TimeStepper_test
  COMMAND TimeStepper_test)
