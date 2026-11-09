# Command line arguments setting one particle per cell for every species listed in Particle.speciesNames of
# INPUT_FILE: the runtests only check that an input is valid, so they use as few particles as possible
function(get_runtest_particle_args INPUT_FILE OUT_VAR)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${INPUT_FILE})
  file(STRINGS ${INPUT_FILE} SPECIES_LINES REGEX "^[ \t]*Particle\\.speciesNames[ \t]*=")
  list(POP_BACK SPECIES_LINES SPECIES_LINE)
  string(REGEX REPLACE "^[^=]*=([^#]*).*$" "\\1" SPECIES "${SPECIES_LINE}")
  separate_arguments(SPECIES UNIX_COMMAND "${SPECIES}")
  list(TRANSFORM SPECIES REPLACE "^.+$" "Particle.\\0.nPartPerCell=2" OUTPUT_VARIABLE ARGS)
  set(${OUT_VAR} ${ARGS} PARENT_SCOPE)
endfunction()

# Stolen from https://gist.github.com/mikedld/a687baabc952a7f061b6a2fd3069f91c
function(add_executable_input_to_tests DIR)
  get_property(TGTS DIRECTORY "${DIR}" PROPERTY BUILDSYSTEM_TARGETS)
  foreach(TGT IN LISTS TGTS)
    get_target_property(target_folder ${TGT} SOURCE_DIR)
    get_target_property(RUN_DIR ${TGT} BINARY_DIR)
    cmake_path(GET target_folder STEM LAST_ONLY MODEL)
    file(GLOB inputs RELATIVE ${target_folder} "${target_folder}/*.input")
    foreach(INPUT IN LISTS inputs)
      get_runtest_particle_args(${target_folder}/${INPUT} PARTICLE_ARGS)
      if(GEMPIC_USE_CUDA OR GEMPIC_USE_HIP) # use one process for GPU pipeline runners with a single logical GPU
        add_test(NAME RunTest.Example.${MODEL}.${INPUT}
          COMMAND mpirun -np 1 -wdir ${RUN_DIR} ${RUN_DIR}/${TGT} ${target_folder}/${INPUT} TimeLoop.nSteps=1 TimeLoop.nmaxSteps=1
          ${PARTICLE_ARGS}
        )
      else()
        add_test(NAME RunTest.Example.${MODEL}.${INPUT}
          COMMAND mpirun -np 2  -wdir ${RUN_DIR} ${RUN_DIR}/${TGT} ${target_folder}/${INPUT}
          TimeLoop.nSteps=1 TimeLoop.nmaxSteps=1
          ReducedDiagnostics.saveFolder=ReducedDiagnostics/${INPUT} # ensure that parallel tests do not write to the same file
          FullDiagnostics.saveFolder=FullDiagnostics/${INPUT}
          ${PARTICLE_ARGS}
        )
      endif()
      # add label to alow running independently from Google tests
      set_tests_properties(
        RunTest.Example.${MODEL}.${INPUT}
        PROPERTIES LABELS "non-Google")
    endforeach()
  endforeach()

  get_property(SUBDIRS DIRECTORY "${DIR}" PROPERTY SUBDIRECTORIES)
  foreach(SUBDIR IN LISTS SUBDIRS)
    add_executable_input_to_tests("${SUBDIR}")
  endforeach()
endfunction()

add_executable_input_to_tests(${PROJECT_SOURCE_DIR}/Examples)
