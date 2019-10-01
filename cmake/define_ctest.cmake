MACRO( _CTEST_FILE_CMP _test )

  ADD_CUSTOM_COMMAND(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${_test}.output
    COMMAND ${_test}
    # lines if it should pipe the console output (if unit test has print instead of writing into a file)
    # > ${CMAKE_CURRENT_BINARY_DIR}/${_test}.screen-output.tmp
    # COMMAND mv ${CMAKE_CURRENT_BINARY_DIR}/${_test}.screen-output.tmp
    # ${CMAKE_CURRENT_BINARY_DIR}/${_test}.output
    DEPENDS ${_test}
    )
  
  ADD_CUSTOM_COMMAND(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${_test}.diff
    COMMAND
    if (${TEST_DIFF} ${CMAKE_CURRENT_SOURCE_DIR}/${_test}.expected_output
        ${CMAKE_CURRENT_BINARY_DIR}/${_test}.output
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
    ${CMAKE_CURRENT_BINARY_DIR}/${_test}.output
    )
  # add the target for this output file to the dependencies of this test
  ADD_CUSTOM_TARGET(${_test}.diff
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${_test}.diff
    )

  #ADD_DEPENDENCIES(_tests ${_test}.diff)
  ADD_TEST(NAME ${_test}
    COMMAND
    ${CMAKE_COMMAND}
    -DBINARY_DIR=${CMAKE_BINARY_DIR}
    -DTESTNAME=${_test}
    -DERROR="Test ${_test} failed"
    -P ${CMAKE_SOURCE_DIR}/cmake/run_test.cmake
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

ENDMACRO()


# Find diff and numdiff for ctests
FIND_PROGRAM(DIFF_EXECUTABLE
  NAMES diff
  HINTS ${DIFF_DIR}
  PATH_SUFFIXES bin
  )

FIND_PROGRAM(NUMDIFF_EXECUTABLE
  NAMES numdiff
  HINTS ${NUMDIFF_DIR}
  PATH_SUFFIXES bin
  )
