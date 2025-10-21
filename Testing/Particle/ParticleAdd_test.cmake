cmake_policy(SET CMP0074 NEW)
find_package(Python3)

execute_process(COMMAND mpirun -np 1 ${TEST_EXECUTABLE}
                RESULT_VARIABLE ERROR_CODE)
if(ERROR_CODE)
  message(FATAL_ERROR "Executable '${TEST_EXECUTABLE}' failed to run!")
endif()
if (Python3_FOUND)
  execute_process(COMMAND ${Python3_EXECUTABLE} ${PYTHON_TEST}
                  RESULT_VARIABLE ERROR_CODE)
  if(ERROR_CODE)
    message(FATAL_ERROR "Test failed!")
  endif()
endif()