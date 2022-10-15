#Check whether sphinx exists on the machine. If it does the job will pass, if not it fails 

find_package(Sphinx PATHS ${CMAKE_SOURCE_DIR}/cmake)
if (SPHINX_FOUND)
	message("Sphinx pages will be built according to CI")
else (SPHINX_FOUND)
	message("Sphinx is not installed")
endif (SPHINX_FOUND)
