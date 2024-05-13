# Version of gempic. Should probably be expanded a bit.
execute_process(COMMAND git fetch origin 'refs/tags/*:refs/tags/*'
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                ERROR_QUIET)
execute_process(COMMAND git describe --tags --dirty
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                OUTPUT_VARIABLE REPO_TAG
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
if (REPO_TAG)
  set(GEMPIC_VERSION "${REPO_TAG}")
else()
  set(GEMPIC_VERSION "Unknown")
endif()