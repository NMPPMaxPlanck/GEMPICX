# Find diff for ctests
find_program(DIFF_EXEC
  NAMES diff
  HINTS ${DIFF_DIR}
  PATH_SUFFIXES bin
  )
  
mark_as_advanced(DIFF_EXEC)

if("${DIFF}" STREQUAL "")
  if(NOT DIFF_EXEC MATCHES "-NOTFOUND")
    set(DIFF ${DIFF_EXEC})
  else()
    message(FATAL_ERROR
      "Could not find diff, which is required for running the test.\n"
      "Please specify DIFF by hand."
      )
  endif()
endif()

execute_process(COMMAND mpirun -np 1 ${PARAMETERS_OUTPUT_TEST} ${INPUT_FILE}
                RESULT_VARIABLE ERROR_CODE)
if(ERROR_CODE)
  message(FATAL_ERROR "Executable ${PARAMETERS_OUTPUT_TEST} failed to run")
endif()
execute_process(COMMAND mpirun -np 1 ${PARAMETERS_OUTPUT_TEST} ${PARAMETERS_OUTPUT_TEST}.output
                RESULT_VARIABLE ERROR_CODE)
if(ERROR_CODE)
  message(FATAL_ERROR "Executable ${PARAMETERS_OUTPUT_TEST} failed to run generated output file")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} -E rename ${PARAMETERS_OUTPUT_TEST}.output
                                                   ${PARAMETERS_OUTPUT_TEST}.old_output 
                RESULT_VARIABLE ERROR_CODE)
if(ERROR_CODE)
  message(FATAL_ERROR "Failed to copy output file")
endif()
execute_process(COMMAND mpirun -np 1 ${PARAMETERS_OUTPUT_TEST} ${PARAMETERS_OUTPUT_TEST}.old_output
                RESULT_VARIABLE ERROR_CODE)
if(ERROR_CODE)
  message(FATAL_ERROR "Executable ${PARAMETERS_OUTPUT_TEST} failed to run second generated output file")
endif()
message(STATUS "Diffing ${PARAMETERS_OUTPUT_TEST}.old_output and ${PARAMETERS_OUTPUT_TEST}.output:")
execute_process(COMMAND ${DIFF} ${PARAMETERS_OUTPUT_TEST}.old_output
                                ${PARAMETERS_OUTPUT_TEST}.output
                RESULT_VARIABLE ERROR_CODE)
if(ERROR_CODE)
  message(FATAL_ERROR "Test failed! ${PARAMETERS_OUTPUT_TEST}.old_output and ${PARAMETERS_OUTPUT_TEST}.output are supposed to be identical.")
endif()
