# based on hyper.deal doxygen

option(GEMPIC_DOCUMENTATION "Build Doxygen documentation" ON)

if(GEMPIC_DOCUMENTATION)
  #check if Doxygen is installed
  find_package(Doxygen)
  if(DOXYGEN_FOUND)
    file(GLOB_RECURSE GEMPIC_PUBLIC_HEADERS ${CMAKE_SOURCE_DIR}/src/*.h)
    
    set(DOXYGEN_INDEX_FILE ${CMAKE_CURRENT_SOURCE_DIR}/html/index.html)
    # set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    #set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    #Replace variables inside @@ with the current values
    #configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)
    #file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR}) #Doxygen won't create this for us
    add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
      DEPENDS ${GEMPIC_PUBLIC_HEADERS}
      COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile.in
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      MAIN_DEPENDENCY Doxyfile.in
      COMMENT "Generating docs")
    add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})
  else(DOXYGEN_FOUND)
    message("Doxygen is not installed")
  endif(DOXYGEN_FOUND)
endif(GEMPIC_DOCUMENTATION)

