# based on hyper.deal doxygen

option(GEMPIC_DOCUMENTATION "Build Doxygen documentation" ON)

if (GEMPIC_DOCUMENTATION)
  # check if Doxygen is installed
  find_package(Doxygen)
  if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    
    # copy input to output file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message("Configured to build Doxygen documentation")
    
    # create documentation
    add_custom_target( doxygen
      COMMAND gsed -i "s|CMAKE_CURRENT_SOURCE_DIR|${CMAKE_CURRENT_SOURCE_DIR}|g" ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
      COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT} > warnings.out 2>&1
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating Doxygen documentation"
      VERBATIM )
  else (DOXYGEN_FOUND)
    message("Doxygen is not installed")
  endif (DOXYGEN_FOUND)
endif (GEMPIC_DOCUMENTATION)
