function(append_testconfig FILE)
  file(APPEND ${FILE} "\n\n")
  file(APPEND ${FILE} "# The following lines overwrite existing values to reduce runtime of example scripts for testing purposes\n")
  file(APPEND ${FILE} "TimeLoop.nSteps = 1\n")
  file(APPEND ${FILE} "ReducedDiagnostics.enable = 0\n")
  file(APPEND ${FILE} "FullDiagnostics.enable = 0\n")
endfunction()

function(select_executable NAME)
  if(NAME STREQUAL "Electrostatic")
    set(EXECUTABLE "electrostatic" PARENT_SCOPE)
  elseif(NAME STREQUAL "ColdPlasma")
    set(EXECUTABLE "cold_plasma" PARENT_SCOPE)
  elseif(NAME STREQUAL "LinVlasovMaxwell")
    set(EXECUTABLE "lin_vlasov_maxwell" PARENT_SCOPE)
  elseif(NAME STREQUAL "VlasovMaxwell")
    set(EXECUTABLE "vlasovmaxwell" PARENT_SCOPE)
  endif()
endfunction()

# Using relative paths to extract the model name
file(GLOB MODELS RELATIVE ${CMAKE_SOURCE_DIR}/Examples "${CMAKE_SOURCE_DIR}/Examples/*")
list(REMOVE_ITEM MODELS "CMakeLists.txt")
foreach(MODEL ${MODELS})
  # Directory with executables where tests are run
  set(RUN_DIR ${CMAKE_BINARY_DIR}/Examples/${MODEL})
  file(GLOB INPUT_FILES RELATIVE ${CMAKE_SOURCE_DIR}/Examples/${MODEL}
    "${CMAKE_SOURCE_DIR}/Examples/${MODEL}/*.input"
    )
  foreach(INPUT_FILE ${INPUT_FILES})
    file(COPY ${CMAKE_SOURCE_DIR}/Examples/${MODEL}/${INPUT_FILE} DESTINATION ${RUN_DIR})
    append_testconfig(${RUN_DIR}/${INPUT_FILE})
    select_executable(${MODEL})
    add_test(NAME RunTest.Example.${MODEL}.${INPUT_FILE}
      COMMAND mpirun -np 2 --oversubscribe -wdir ${RUN_DIR} ${RUN_DIR}/${EXECUTABLE} ${RUN_DIR}/${INPUT_FILE}
      )
  endforeach()
endforeach()
