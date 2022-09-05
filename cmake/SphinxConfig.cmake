find_program(
    SPHINX_EXECUTABLE
    NAMES sphinx-build
    HINTS $ENV{SPHINX_DIR})
mark_as_advanced(SPHINX_EXECUTABLE)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Sphinx "Failed to locate sphinx-build executable" SPHINX_EXECUTABLE)
