# based on hyper.deal doxygen

option(GEMPIC_DOCUMENTATION "Build Doxygen documentation" ON)

#if (GEMPIC_DOCUMENTATION)
#  # check if Doxygen is installed
#  find_package(Doxygen)
#  if (DOXYGEN_FOUND)
#    # set input and output files
#    set(DOXYGEN_IN Doxyfile.in)
#    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
#    
#    # copy input to output file
#    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
#    message("Configured to build Doxygen documentation")
#    
#    # create documentation
#    add_custom_target( doxygen
#      COMMAND gsed -i "s|CMAKE_CURRENT_SOURCE_DIR|${CMAKE_CURRENT_SOURCE_DIR}|g" ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
#      COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT} > warnings.out 2>&1
#      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#      COMMENT "Generating Doxygen documentation"
#      VERBATIM )
#  else (DOXYGEN_FOUND)
#    message("Doxygen is not installed")
#  endif (DOXYGEN_FOUND)
#endif (GEMPIC_DOCUMENTATION)

#if (GEMPIC_DOCUMENTATION)
#    #check if Doxygen is installed
#    find_package(Doxygen)
#    if (DOXYGEN_FOUND)
#        #get_target_property(GEMPIC_PUBLIC_HEADER_DIR gempic INTERFACE_INCLUDE_DIRECTORIES)
#        file(GLOB_RECURSE GEMPIC_PUBLIC_HEADERS ${CMAKE_SOURCE_DIR}/src/*.h)
#        
#        set(DOXYGEN_INPUT_DIR ${CMAKE_SOURCE_DIR}/src)
#        set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
#        set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)
#        set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
#        set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
#
#        #Replace variables inside @@ with the current values
#        configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)
#
#        file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR}) #Doxygen won't create this for us
#        add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
#                           DEPENDS ${GEMPIC_PUBLIC_HEADERS}
#                           COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
#                           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#                           MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
#                           COMMENT "Generating docs")
#
#        add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})
#    else (DOXYGEN_FOUND)
#        message("Doxygen is not installed")
#    endif (DOXYGEN_FOUND)
#endif (GEMPIC_DOCUMENTATION)

if (GEMPIC_DOCUMENTATION)
    #check if Doxygen is installed
    find_package(Doxygen)
    if (DOXYGEN_FOUND)
        #get_target_property(GEMPIC_PUBLIC_HEADER_DIR gempic INTERFACE_INCLUDE_DIRECTORIES)
        file(GLOB_RECURSE GEMPIC_PUBLIC_HEADERS ${CMAKE_SOURCE_DIR}/src/*.h)
        
        #set(DOXYGEN_INPUT_DIR ${CMAKE_SOURCE_DIR}/src)
        #set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
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
    else (DOXYGEN_FOUND)
        message("Doxygen is not installed")
    endif (DOXYGEN_FOUND)
endif (GEMPIC_DOCUMENTATION)

