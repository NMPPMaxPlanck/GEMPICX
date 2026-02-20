target_sources(GtestTests PRIVATE
    # test cases
    RKButcherTableau_test.cpp
    RungeKutta_test.cpp
    SplittingColdPlasma_test.cpp
    TimeLoop_test.cpp
    TimeLoopOrder_test.cpp
    )

add_executable(
  TimeStepper_test
  TimeStepper_test.cpp)

target_link_libraries(
  TimeStepper_test
  GEMPICX_TIMESTEPPER)

add_test(
  NAME TS_test
  COMMAND TimeStepper_test TSTest00)
