function(confirm_tag _name _location) #GIT_TAG (optional argument)
  # Checks that a folder at _location
  # 1) Exists and is a git repository
  # 2) If given, that the extra argument corresponds to tag name of the repository
  # Check for optional args:
  set(oneValueArgs GIT_TAG)
  cmake_parse_arguments("WANTED" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
  # Check if we're in a (git repo) folder
  execute_process(COMMAND git rev-parse --show-toplevel
                  WORKING_DIRECTORY ${_location}
                  OUTPUT_VARIABLE REPO_TOP_LEVEL
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET)
  execute_process(COMMAND git rev-parse --show-toplevel
                  OUTPUT_VARIABLE REPO_GEMPIC_TOP_LEVEL
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET)
  if ("${REPO_TOP_LEVEL}" STREQUAL "" OR "${REPO_TOP_LEVEL}" STREQUAL "${REPO_GEMPIC_TOP_LEVEL}")
    message(STATUS "${_location} does not exist, or is not a separate git repository.")
    set(${_name}_CORRECT_VERSION_FOUND False PARENT_SCOPE)
    return()
  endif()
  
  execute_process(COMMAND git describe --tags --dirty
                  WORKING_DIRECTORY ${_location}
                  OUTPUT_VARIABLE REPO_TAG
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET)

  # Repositories with tags yield non-empty outputs
  if (REPO_TAG)
    set(TAGGED_REPO TRUE)
    # git describe --tags --dirty gives you a string consisting of
    # The nearest tag name: "([^-]*)"
    # (optionally, if you're not directly on a tag) The number of commits since then: "(-[0-9]+)?"
    # (optionally, if you're not directly on a tag) The shortened SHA of the branch, prefaced with g
    # (optionally, if the repo is dirty) ...-dirty (not searched for)
    string(REGEX MATCH "([^-]*)(-[0-9]+)?(-g([^-]*))?" _ "${REPO_TAG}")
  # Repositories with no tags need --always
  else()
    execute_process(COMMAND git describe --dirty --always
                    WORKING_DIRECTORY ${_location}
                    OUTPUT_VARIABLE REPO_TAG
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    )
    # git describe --dirty --always gives you a string consisting of
    # The shortened SHA of the branch
    # (optionally, if the repo is dirty) ...-dirty (not searched for)
    string(REGEX MATCH "([^-]*)" _ "${REPO_TAG}")
  endif()

  execute_process(COMMAND git rev-parse "${CMAKE_MATCH_0}"
                  WORKING_DIRECTORY ${_location}
                  OUTPUT_VARIABLE REPO_COMMIT_ID
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  )

  if (TAGGED_REPO AND ${CMAKE_MATCH_COUNT} EQUAL 1)
    set(VERSION_NAME "Version ${CMAKE_MATCH_0}")
  else()
    set(VERSION_NAME "Commit ${REPO_COMMIT_ID}")
  endif()
  if(${REPO_TAG} MATCHES "-dirty")
    set(VERSION_NAME "${VERSION_NAME} (dirty)")
  endif()

  message(STATUS "Found ${_name}: ${${_name}_SOURCE_DIR} (${VERSION_NAME})")

  # We check if the repository matches our requirements
  if (DEFINED WANTED_GIT_TAG)
    if (${REPO_TAG} MATCHES "-dirty")
      message(STATUS "... but the repository is dirty.")
      message(FATAL_ERROR "To use dirty repositories, don't specify a GIT_TAG.")
      set(${_name}_CORRECT_VERSION_FOUND false PARENT_SCOPE)
    else()
      execute_process(COMMAND git rev-parse "${WANTED_GIT_TAG}"
                      WORKING_DIRECTORY ${_location}
                      OUTPUT_VARIABLE WANTED_COMMIT_ID
                      OUTPUT_STRIP_TRAILING_WHITESPACE
                      ERROR_QUIET
                      )
      if("${WANTED_COMMIT_ID}" STREQUAL "${REPO_COMMIT_ID}")
        set(${_name}_CORRECT_VERSION_FOUND true PARENT_SCOPE)
      else()
        message(STATUS "... which is the wrong version.")
        set(${_name}_CORRECT_VERSION_FOUND false PARENT_SCOPE)
      endif()
    endif()
  else()
    set(${_name}_CORRECT_VERSION_FOUND true PARENT_SCOPE)
  endif()
endfunction()

# A modification on the FetchContent_Declare command, trying to avoid redownloading dependencies
# every time the build is reconfigured, and allow the SOURCE_DIR argument of FetchContent_Declare
# to be set to a non-default directory at the same time.
# Takes one mandatory command and three optional ones:
#
#   _name:          The name of the library to fetch
#   SOURCE_DIR:     The location of said library.
#                   A search is first done to see if the library already exists in this location.
#   GIT_REPOSITORY: The git address of the library
#   GIT_TAG:        The tag or commit ID to get. If given, and a library is already found,
#                   the downloaded library is checked against the tag/commit ID.
#   ...             Any other arguments are passed on to FetchContent_Declare unmodified.
macro(gempic_FetchContent_Declare _name)
  set(oneValueArgs GIT_REPOSITORY GIT_TAG SOURCE_DIR)
  cmake_parse_arguments(${_name} "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  message(STATUS "Searching for ${_name} installs ...")
  find_package(${_name} CONFIG)

  if (NOT ${${_name}_FOUND})
    message(STATUS "Searching for ${_name} source folders ...")
    if (NOT DEFINED ${_name}_SOURCE_DIR)
      string(TOLOWER ${_name} _name_lower)
      set(${_name}_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third_party/${_name_lower}-src/")
      message(STATUS "No SOURCE_DIR provided for ${_name}. Assuming: ${${_name}_SOURCE_DIR}")
    endif()

    string(TOUPPER "${_name}" ${_name}_UPPERCASE)

    # Find the library
    if(DEFINED ${_name}_GIT_TAG)
      set(${_name}_GIT_TAG_LINE GIT_TAG ${${_name}_GIT_TAG})
    endif()
    confirm_tag(${_name} ${${_name}_SOURCE_DIR} ${${_name}_GIT_TAG_LINE})
    if (${${_name}_CORRECT_VERSION_FOUND})
      set(FETCHCONTENT_SOURCE_DIR_${${_name}_UPPERCASE} ${${_name}_SOURCE_DIR})

      # Strictly speaking unnecessary, because the library source was found
      set(${_name}_GIT_REPOSITORY_LINE)
      set(${_name}_GIT_TAG_LINE)
      set(${_name}_GIT_UPDATE_LINE)
    else()
      unset(FETCHCONTENT_SOURCE_DIR_${${_name}_UPPERCASE} CACHE)
      unset(FETCHCONTENT_SOURCE_DIR_${${_name}_UPPERCASE})
      # Library wasn't found
      if(NOT DEFINED ${_name}_GIT_REPOSITORY)
        message(FATAL_ERROR "gempic_fetch() requires a GIT_REPOSITORY when no matching library is found.")
      else()
        set(${_name}_GIT_REPOSITORY_LINE GIT_REPOSITORY ${${_name}_GIT_REPOSITORY})
      endif()
      message(STATUS "No suitable source folders found. Fetching ${_name} from ${${_name}_GIT_REPOSITORY} ...")
      # Ensure repository is updated (CHECKOUT) and changes are not permanently deleted (REBASE_)
      # See https://cmake.org/cmake/help/latest/module/ExternalProject.html#git
      #set(${_name}_GIT_UPDATE_LINE GIT_REMOTE_UPDATE_STRATEGY CHECKOUT)
    endif()

    FetchContent_Declare(
      ${_name}
      ${${_name}_GIT_REPOSITORY_LINE}
      ${${_name}_GIT_TAG_LINE}
      ${${_name}_GIT_UPDATE_LINE}
      SOURCE_DIR ${${_name}_SOURCE_DIR}
      ${${_name}_UNPARSED_ARGUMENTS}  # Give remaining arguments to FetchContent_Declare()
    )
  endif()
endmacro()
