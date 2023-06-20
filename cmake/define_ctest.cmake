macro( _CTEST_FILE_CMP _test )

  if(EXISTS  ${CMAKE_CURRENT_SOURCE_DIR}/${_test}.input )
    message("Input file ${_test}")
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${_test}.code_output
      COMMAND mpirun -np 4 --oversubscribe ../${_test} ${CMAKE_CURRENT_SOURCE_DIR}/${_test}.input
      COMMAND tail -n +2 ${CMAKE_CURRENT_BINARY_DIR}/${_test}.output > ${CMAKE_CURRENT_BINARY_DIR}/${_test}.code_output
      # lines if it should pipe the console output (if unit test has print instead of writing into a file)
      # > ${CMAKE_CURRENT_BINARY_DIR}/${_test}.screen-output.tmp
      # COMMAND mv ${CMAKE_CURRENT_BINARY_DIR}/${_test}.screen-output.tmp
      # ${CMAKE_CURRENT_BINARY_DIR}/${_test}.output
      DEPENDS ${_test}
      )
  else()
    message("No input file ${_test}")
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${_test}.code_output
      COMMAND mpirun -np 4 --oversubscribe ../${_test}
      COMMAND tail -n +2 ${CMAKE_CURRENT_BINARY_DIR}/${_test}.output > ${CMAKE_CURRENT_BINARY_DIR}/${_test}.code_output
      # lines if it should pipe the console output (if unit test has print instead of writing into a file)
      # > ${CMAKE_CURRENT_BINARY_DIR}/${_test}.screen-output.tmp
      # COMMAND mv ${CMAKE_CURRENT_BINARY_DIR}/${_test}.screen-output.tmp
      # ${CMAKE_CURRENT_BINARY_DIR}/${_test}.output
      DEPENDS ${_test}
      )
  endif()

  
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${_test}.diff
    COMMAND
    if (${TEST_DIFF} ${CMAKE_CURRENT_SOURCE_DIR}/${_test}.expected_output
        ${CMAKE_CURRENT_BINARY_DIR}/${_test}.code_output
        > ${CMAKE_CURRENT_BINARY_DIR}/${_test}.diff) \; then
    : \;
    else
    mv ${CMAKE_CURRENT_BINARY_DIR}/${_test}.diff
    ${CMAKE_CURRENT_BINARY_DIR}/${_test}.diff.failed \;
    echo "******* Error during diffing output results for ${_test}" \;
    echo "******* Results are stored in ${CMAKE_CURRENT_BINARY_DIR}/${_test}.diff.failed" \;
    echo "******* Check: ${CMAKE_CURRENT_BINARY_DIR}/${_test} ${CMAKE_CURRENT_SOURCE_DIR}/${_test}" \;
    echo "******* Diffs are:" \;
    cat ${CMAKE_CURRENT_BINARY_DIR}/${_test}.diff.failed \;
    false \;
    fi
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${_test}.expected_output
    ${CMAKE_CURRENT_BINARY_DIR}/${_test}.code_output
    )
  # add the target for this output file to the dependencies of this test
  add_custom_target(${_test}.diffFile
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${_test}.diff
    )

  add_test(NAME ${_test}
    COMMAND
    ${CMAKE_COMMAND}
    -DBINARY_DIR=${CMAKE_BINARY_DIR}
    -DTESTNAME=${_test}
    -DERROR="Test ${_test} failed"
    -P ${CMAKE_SOURCE_DIR}/cmake/run_test.cmake
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

endmacro()


# Find diff and numdiff for ctests
find_program(DIFF_EXECUTABLE
  NAMES diff
  HINTS ${DIFF_DIR}
  PATH_SUFFIXES bin
  )

find_program(NUMDIFF_EXECUTABLE
  NAMES numdiff
  HINTS ${NUMDIFF_DIR}
  PATH_SUFFIXES bin
  )
  
mark_as_advanced(DIFF_EXECUTABLE NUMDIFF_EXECUTABLE)

if("${TEST_DIFF}" STREQUAL "")
  if(NOT NUMDIFF_EXECUTABLE MATCHES "-NOTFOUND")
    set(TEST_DIFF ${NUMDIFF_EXECUTABLE} -a 1e-5 -r 1e-8 -s ' \\t\\n:,')
    if(DIFF_EXECUTABLE MATCHES "-NOTFOUND")
      set(DIFF_EXECUTABLE ${NUMDIFF_EXECUTABLE})
    endif()
  elseif(NOT DIFF_EXECUTABLE MATCHES "-NOTFOUND")
    set(TEST_DIFF ${DIFF_EXECUTABLE})
    message(
      "######### Could not find numdiff. This will cause a number of ctests to fail. \n"
      )
  else()
    message(FATAL_ERROR
      "Could not find diff or numdiff. One of those are required for running the tests.\n"
      "Please specify TEST_DIFF by hand."
      )
  endif()
endif()
##############
