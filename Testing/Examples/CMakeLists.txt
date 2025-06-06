# Stolen from https://gist.github.com/mikedld/a687baabc952a7f061b6a2fd3069f91c
function(add_executable_input_to_tests DIR)
  get_property(TGTS DIRECTORY "${DIR}" PROPERTY BUILDSYSTEM_TARGETS)
  foreach(TGT IN LISTS TGTS)
    get_target_property(target_folder ${TGT} SOURCE_DIR)
    get_target_property(RUN_DIR ${TGT} BINARY_DIR)
    cmake_path(GET target_folder STEM LAST_ONLY MODEL)
    file(GLOB inputs RELATIVE ${target_folder} "${target_folder}/*.input")
    foreach(INPUT IN LISTS inputs)
      if(GEMPIC_USE_CUDA OR GEMPIC_USE_HIP) # use one process for GPU pipeline runners with a single logical GPU
        add_test(NAME RunTest.Example.${MODEL}.${INPUT}
          COMMAND mpirun -np 1 -wdir ${RUN_DIR} ${RUN_DIR}/${TGT} ${target_folder}/${INPUT} TimeLoop.nSteps=1
        )
      else()
        add_test(NAME RunTest.Example.${MODEL}.${INPUT}
          COMMAND mpirun -np 2 --oversubscribe -wdir ${RUN_DIR} ${RUN_DIR}/${TGT} ${target_folder}/${INPUT}
          TimeLoop.nSteps=1
          ReducedDiagnostics.saveFolder=ReducedDiagnostics/${INPUT} # ensure that parallel tests do not write to the same file
          FullDiagnostics.saveFolder=FullDiagnostics/${INPUT}
        )
      endif()
    endforeach()
  endforeach()

  get_property(SUBDIRS DIRECTORY "${DIR}" PROPERTY SUBDIRECTORIES)
  foreach(SUBDIR IN LISTS SUBDIRS)
    add_executable_input_to_tests("${SUBDIR}")
  endforeach()
endfunction()

add_executable_input_to_tests(${CMAKE_SOURCE_DIR}/Examples)
