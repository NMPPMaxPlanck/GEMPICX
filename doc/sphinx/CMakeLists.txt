if(GEMPIC_DOCUMENTATION AND DOXYGEN_FOUND)
  list(APPEND RST_FILES MaxwellDiscreteForms.rst Examples.rst DeRhamComplexes.rst Tests.rst CodeUnits.rst FormsAndOperators.rst ParticleMesh.rst TimeLoops.rst GoogleTest.rst CodeStyle.rst InputFiles.rst ContributingToDoc.rst Diagnostics.rst)

  # Search for Python executables, starting with local environments first
  cmake_policy(SET CMP0094 NEW)
  find_package(Python3)
  
  #Check whether sphinx exists on the machine. If it does the job will pass, if not it fails 
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sphinx;import breathe"
    RESULT_VARIABLE SPHINX_AND_BREATHE_NOT_FOUND
    ERROR_VARIABLE PYTHON_MODULE_ERROR
    ECHO_ERROR_VARIABLE
  )

  if(SPHINX_AND_BREATHE_NOT_FOUND)
    message(WARNING "${PYTHON_MODULE_ERROR}\nPython3 found: ${Python3_EXECUTABLE}")
    message(WARNING "Sphinx, Breathe or Python is not installed. Building the documentation is not possible. If you have a virtual environment, did you activate it?")
  else()
    set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
    set(SPHINX_BUILD ${CMAKE_BINARY_DIR}/documentation)
    set(LATEX_INPUT ${SPHINX_SOURCE}/latex)
    set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)
    set(SPHINX_GRAPHICS ${SPHINX_SOURCE}/_static/graphics)

    # Download graphics from MPCDF Datashare
    add_custom_command(OUTPUT ${SPHINX_GRAPHICS}.zip ${SPHINX_GRAPHICS}
      COMMAND ${Python3_EXECUTABLE} ${SPHINX_SOURCE}/downloadAndExtractGraphics.py
      DEPENDS ${SPHINX_SOURCE}/downloadAndExtractGraphics.py
      )

    foreach(RSTF ${RST_FILES})
      string(REGEX REPLACE "[.]rst$" ".tex" LATEXF ${RSTF})
      add_custom_command(OUTPUT ${LATEX_INPUT}/${RSTF}
        DEPENDS ${LATEX_INPUT}/${LATEXF} ${SPHINX_GRAPHICS}
        COMMAND pandoc ${LATEXF} --mathjax --columns=120 --wrap=preserve --bibliography GempicReferences.bib --citeproc -o ${RSTF}
        COMMAND ${Python3_EXECUTABLE} convertEq.py ${RSTF}
        VERBATIM

        WORKING_DIRECTORY ${LATEX_INPUT}
      )
    endforeach()
    
    list(TRANSFORM RST_FILES PREPEND ${LATEX_INPUT}/)
    list(APPEND DOCUMENTATION_DEPENDENCIES ${RST_FILES} ${SPHINX_SOURCE}/index.rst Doxygen ${DOXYGEN_INDEX_FILE})

    add_custom_command(OUTPUT ${SPHINX_INDEX_FILE}
      COMMAND ${Python3_EXECUTABLE} ${SPHINX_SOURCE}/callSphinxDuplicateNamespaceWarningSilenced.py
              -m sphinx -b html
              # Tell Breathe where to find the Doxygen output
              -Dbreathe_projects.gempic=${DOXYGEN_OUTPUT_DIR}
              ${SPHINX_SOURCE} ${SPHINX_BUILD}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      DEPENDS ${DOCUMENTATION_DEPENDENCIES}
      MAIN_DEPENDENCY ${SPHINX_SOURCE}/conf.py
      COMMENT "Generating documentation with Sphinx"
    )

    add_custom_target(doc ALL DEPENDS ${SPHINX_INDEX_FILE})
    message(STATUS "Documentation can be built with the command `cmake --build ${CMAKE_BINARY_DIR} --target doc`")
  endif()
endif()