#Check whether sphinx exists on the machine. If it does the job will pass, if not it fails 

find_package(Sphinx PATHS ${CMAKE_SOURCE_DIR}/cmake)
if (SPHINX_FOUND)
	message("Sphinx pages will be built according to CI")
	ADD_CUSTOM_COMMAND(OUTPUT ${CMAKE_SOURCE_DIR}/doc/sphinx/latex/${_file}.rst
	COMMAND pandoc ${_file}.tex --mathjax --wrap=preserve --bibliography spectral_gempic.bib --citeproc -o ${_file}.rst
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/doc/sphinx/latex/
else (SPHINX_FOUND)
	message("Sphinx is not installed")
endif (SPHINX_FOUND)
