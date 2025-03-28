############################################################################################
#
# Preprocess and compile documentation:
# 1. Download and extract files needed by the documentation but not stored in the repository
# 2. Build doxygen documentation
# 3. Convert tex files to rst using pandoc. (Should be automated.)
# 4. Build the final documentation using python with sphinx.
#
############################################################################################
cmake_policy(SET CMP0094 NEW)
set(GEMPIC_DOC_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
find_package(Python3 REQUIRED)

set(GEMPIC_EXTERNAL_DOC_DATA ${GEMPIC_DOC_SOURCE}/_static/graphics)
add_custom_command(OUTPUT ${GEMPIC_EXTERNAL_DOC_DATA}
  COMMAND ${Python3_EXECUTABLE} ${GEMPIC_DOC_SOURCE}/downloadAndExtractGraphics.py
  DEPENDS ${GEMPIC_DOC_SOURCE}/downloadAndExtractGraphics.py
  )


find_package(Doxygen REQUIRED)
file(GLOB_RECURSE GEMPIC_PUBLIC_HEADERS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../Src/*.H)
set(GEMPIC_DOXYGEN_WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(GEMPIC_DOXYGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/doxygen)
set(GEMPIC_DOXYGEN_INDEX_FILE ${GEMPIC_DOXYGEN_OUTPUT_DIR}/index.xml)

add_custom_command(OUTPUT ${GEMPIC_DOXYGEN_INDEX_FILE}
  COMMAND DOXYGEN_OUTPUT_DIR=${GEMPIC_DOXYGEN_OUTPUT_DIR} ${DOXYGEN_EXECUTABLE} Doxyfile
  WORKING_DIRECTORY ${GEMPIC_DOXYGEN_WORKING_DIR}
  DEPENDS ${GEMPIC_PUBLIC_HEADERS} ${GEMPIC_EXTERNAL_DOC_DATA}
  MAIN_DEPENDENCY Doxyfile
)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import sphinx;import breathe"
  RESULT_VARIABLE SPHINX_AND_BREATHE_NOT_FOUND
  ERROR_VARIABLE PYTHON_MODULE_ERROR
  ECHO_ERROR_VARIABLE
)

if(SPHINX_AND_BREATHE_NOT_FOUND)
  message(FATAL_ERROR 
  "Sphinx and/or Breathe not found with ${Python3_EXECUTABLE}. If you have a virtual environment, did you activate it?\n"
  "${Python3_EXECUTABLE} -c 'import sphinx; import breathe'\n"
  "${PYTHON_MODULE_ERROR}\n"
  )
endif()

set(GEMPIC_LATEX_INPUT ${GEMPIC_DOC_SOURCE}/latex)
# ToDo: Automatically detect file which are to be converted with pandoc
# Opt out if a file should not be used.
list(APPEND GEMPIC_RST_TEX_FILES MaxwellDiscreteForms.rst Examples.rst DeRhamComplexes.rst CodeUnits.rst FormsAndOperators.rst ParticleMesh.rst TimeLoops.rst GoogleTest.rst CodeStyle.rst InputFiles.rst ContributingToDoc.rst Diagnostics.rst CodeEditorVSCode.rst)
foreach(RSTF ${GEMPIC_RST_TEX_FILES})
  string(REGEX REPLACE "[.]rst$" ".tex" LATEXF ${RSTF})
  add_custom_command(OUTPUT ${GEMPIC_LATEX_INPUT}/${RSTF}
    COMMAND pandoc ${LATEXF} --mathjax --columns=120 --wrap=preserve --bibliography GempicReferences.bib --citeproc -o ${RSTF}
    COMMAND ${Python3_EXECUTABLE} convertEq.py ${RSTF}
    WORKING_DIRECTORY ${GEMPIC_LATEX_INPUT}
    DEPENDS ${GEMPIC_LATEX_INPUT}/${LATEXF} ${GEMPIC_EXTERNAL_DOC_DATA}
    VERBATIM
  )
endforeach()
list(TRANSFORM GEMPIC_RST_TEX_FILES PREPEND ${GEMPIC_LATEX_INPUT}/)

set(GEMPIC_SPHINX_BUILD ${CMAKE_BINARY_DIR}/sphinx)
set(GEMPIC_SPHINX_INDEX_FILE ${GEMPIC_SPHINX_BUILD}/index.html)
add_custom_command(OUTPUT ${GEMPIC_SPHINX_INDEX_FILE}
  COMMAND ${Python3_EXECUTABLE} ${GEMPIC_DOC_SOURCE}/callSphinxDuplicateNamespaceWarningSilenced.py
          -m sphinx -b html
          -Dbreathe_projects.gempic=${GEMPIC_DOXYGEN_OUTPUT_DIR}
          ${GEMPIC_DOC_SOURCE} ${GEMPIC_SPHINX_BUILD}
  WORKING_DIRECTORY ${GEMPIC_DOC_SOURCE}
  DEPENDS ${GEMPIC_RST_TEX_FILES} ${GEMPIC_DOC_SOURCE}/index.rst ${GEMPIC_DOXYGEN_INDEX_FILE}
  MAIN_DEPENDENCY ${GEMPIC_DOC_SOURCE}/conf.py
  VERBATIM
)
add_custom_target(doc ALL DEPENDS ${GEMPIC_SPHINX_INDEX_FILE})

message(STATUS "Documentation can be built with the command `cmake --build ${CMAKE_BINARY_DIR} --target doc`")
