#Check whether sphinx exists on the machine. If it does the job will pass, if not it fails 

#find_package(Sphinx PATHS ${CMAKE_SOURCE_DIR}/cmake)
#if (SPHINX_FOUND)
#    set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
#    set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/doc/sphinx)
#    set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)
#    add_custom_command(OUTPUT ${SPHINX_INDEX_FILE}
#                       COMMAND 
#                         ${SPHINX_EXECUTABLE} -b html
#                         # Tell Breathe where to find the Doxygen output
#                         -Dbreathe_projects.CatCutifier=${DOXYGEN_OUTPUT_DIR}/xml
#                       ${SPHINX_SOURCE} ${SPHINX_BUILD}
#                       WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#                       DEPENDS
#                       # Other docs files you want to track should go here (or in some variable)
#                       ${CMAKE_CURRENT_SOURCE_DIR}/index.rst
#                       ${DOXYGEN_INDEX_FILE}
#                       MAIN_DEPENDENCY ${SPHINX_SOURCE}/conf.py
#                       COMMENT "Generating documentation with Sphinx")
#
#    # Nice named target so we can run the job easily
#    add_custom_target(Sphinx ALL DEPENDS ${SPHINX_INDEX_FILE})
#	message("Sphinx pages will be built according to CI")
#else (SPHINX_FOUND)
#	message("Sphinx is not installed")
#endif (SPHINX_FOUND)
