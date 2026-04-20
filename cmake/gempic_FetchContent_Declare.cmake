include(FetchContent)
include(cmake/gempic_utils.cmake)

function(gempic_confirm_git_tag)# NAME LOCATION (mandatory) #GIT_TAG ALLOW_DIRTY (optional)
  # Checks that a folder at LOCATION
  # 1) Exists and is a git repository
  # 2) If given, that the extra argument corresponds to tag name of the repository
  #                                               OR to the shortened SHA of the commit
  # Check for optional args:
  #  GIT_TAG:     Specific version to check against
  #  ALLOW_DIRTY: Checks the GIT_TAG version if given, but allows user modified repos.
  set(options ALLOW_DIRTY)
  set(mandatoryArgs NAME LOCATION)
  set(oneValueArgs ${mandatoryArgs} GIT_TAG)
  cmake_parse_arguments("wanted" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  gempic_check_required_variables(CHECK_VARIABLES ${mandatoryArgs} PREFIX wanted)
  
  # Check if we're in a (git repo) folder
  execute_process(COMMAND git rev-parse --show-toplevel
                  WORKING_DIRECTORY ${wanted_LOCATION}
                  OUTPUT_VARIABLE REPO_TOP_LEVEL
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET)
  execute_process(COMMAND git rev-parse --show-toplevel
                  OUTPUT_VARIABLE REPO_GEMPIC_TOP_LEVEL
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET)
  if("${REPO_TOP_LEVEL}" STREQUAL "" OR "${REPO_TOP_LEVEL}" STREQUAL "${REPO_GEMPIC_TOP_LEVEL}")
    message(STATUS "${wanted_LOCATION} does not exist, or is not a separate git repository.")
    set(${wanted_NAME}_CORRECT_VERSION_FOUND FALSE PARENT_SCOPE)
    return()
  endif()
  
  execute_process(COMMAND git describe --tags --dirty
                  WORKING_DIRECTORY ${wanted_LOCATION}
                  OUTPUT_VARIABLE REPO_TAG
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET)

  # Repositories with tags yield non-empty outputs
  if(REPO_TAG)
    set(TAGGED_REPO TRUE)
    # git describe --tags --dirty gives you a string consisting of
    # The nearest tag name: "([^-]*)"
    # (optionally, if you're not directly on a tag) The number of commits since then: "(-[0-9]+)?"
    # (optionally, if you're not directly on a tag) The shortened SHA of the commit, prefaced with g
    # (optionally, if the repo is dirty) ...-dirty (not searched for)
    string(REGEX MATCH "([^-]*)(-[0-9]+)?(-g([^-]*))?" _ "${REPO_TAG}")
  # Repositories with no tags need --always
  else()
    execute_process(COMMAND git describe --dirty --always
                    WORKING_DIRECTORY ${wanted_LOCATION}
                    OUTPUT_VARIABLE REPO_TAG
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    )
    # git describe --dirty --always gives you a string consisting of
    # The shortened SHA of the commit
    # (optionally, if the repo is dirty) ...-dirty (not searched for)
    string(REGEX MATCH "([^-]*)" _ "${REPO_TAG}")
  endif()

  execute_process(COMMAND git rev-parse "${CMAKE_MATCH_0}"
                  WORKING_DIRECTORY ${wanted_LOCATION}
                  OUTPUT_VARIABLE REPO_COMMIT_ID
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  )

  if(TAGGED_REPO AND ${CMAKE_MATCH_COUNT} EQUAL 1)
    set(VERSION_NAME "Version ${CMAKE_MATCH_0}")
  else()
    set(VERSION_NAME "Commit ${REPO_COMMIT_ID}")
  endif()
  if(${REPO_TAG} MATCHES "-dirty")
    set(VERSION_NAME "${VERSION_NAME} (dirty)")
  endif()

  message(STATUS "Found ${wanted_NAME}: ${${wanted_NAME}_SOURCE_DIR} (${VERSION_NAME})")
  
  set(${wanted_NAME}_VERSION_FOUND ${VERSION_NAME} PARENT_SCOPE)
  # We check if the repository matches our requirements
  if(DEFINED wanted_GIT_TAG)
    execute_process(COMMAND git rev-parse "${wanted_GIT_TAG}"
                    WORKING_DIRECTORY ${wanted_LOCATION}
                    OUTPUT_VARIABLE wanted_COMMIT_ID
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET
                    )
    if("${wanted_COMMIT_ID}" STREQUAL "${REPO_COMMIT_ID}")
      set(STATUS_MESSAGE "${wanted_NAME} version matches")
      set(${wanted_NAME}_CORRECT_VERSION_FOUND TRUE PARENT_SCOPE)
      if("${REPO_TAG}" MATCHES "-dirty")
        if(wanted_ALLOW_DIRTY)
          message(WARNING "${STATUS_MESSAGE} (Dirty version of repository specifically allowed).")
        else()
          message(STATUS "${STATUS_MESSAGE} ... but the repository is dirty.")
          set(${wanted_NAME}_CORRECT_VERSION_FOUND FALSE PARENT_SCOPE)
          message(FATAL_ERROR "To use dirty repositories, don't specify a GIT_TAG, or toggle the ALLOW_DIRTY option.")
        endif()
      else()
        message(STATUS "${STATUS_MESSAGE}!")
      endif()
    else()
      message(STATUS "... which is the wrong version.")
      set(${wanted_NAME}_CORRECT_VERSION_FOUND FALSE PARENT_SCOPE)
    endif()
  else()
    set(${wanted_NAME}_CORRECT_VERSION_FOUND TRUE PARENT_SCOPE)
  endif()
endfunction()

# A modification on the FetchContent_Declare command, trying to avoid redownloading dependencies
# every time the build is reconfigured, and allow the SOURCE_DIR argument of FetchContent_Declare
# to be set to a non-default directory at the same time.
# Takes one mandatory command and three optional ones:
#
#   NAME:           The name of the library to fetch
#   SOURCE_DIR:     The location of said library.
#                   A search is first done to see if the library already exists in this location.
#   GIT_REPOSITORY: The git address of the library
#   GIT_TAG:        The tag or commit ID to get. If given, and a library is already found,
#                   the downloaded library is checked against the tag/commit ID.
#   ALLOW_DIRTY:    Downloads the GIT_TAG version if given, but allows the user to modify the repo.
#   ...             Any other arguments are passed on to FetchContent_Declare unmodified.
function(gempic_FetchContent_Declare NAME)
  set(oneValueArgs GIT_REPOSITORY GIT_TAG SOURCE_DIR ALLOW_DIRTY)
  set(multiValueArgs FIND_PACKAGE_ARGS)
  cmake_parse_arguments(arg "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  message(STATUS "Searching for ${NAME} installs ...")
  if(NOT DEFINED ${NAME}_SOURCE_DIR)
    set(${NAME}_SOURCE_DIR ${arg_SOURCE_DIR})
  endif()
  find_package(${NAME} CONFIG ${arg_FIND_PACKAGE_ARGS})

  if(${${NAME}_FOUND})
    message(STATUS "Found ${NAME} install: ${${NAME}_DIR}")
    if(${NAME}_VERSION)
      set(GEMPICX_${NAME}_VERSION ${${NAME}_VERSION} CACHE STRING
          "Version set by find_package" FORCE)
      if(${arg_GIT_TAG} MATCHES ${${NAME}_VERSION})
        set(${NAME}_CORRECT_VERSION_INSTALLED TRUE)
      endif()
    else()
      unset(GEMPICX_${NAME}_VERSION CACHE)
      unset(GEMPICX_${NAME}_VERSION)
    endif()
    if(arg_GIT_TAG)
      if(${${NAME}_CORRECT_VERSION_INSTALLED})
        message(STATUS "${NAME} version matches")
      else()
        message(WARNING "No version compatibility check is executed for the installed package: "
                "'${NAME}'. Proceed at your own risk.")
      endif()
    endif()
  else()
    message(STATUS "Searching for ${NAME} source folders ...")
    if(NOT DEFINED ${NAME}_SOURCE_DIR)
      string(TOLOWER ${NAME} NAME_LOWER)
      set(${NAME}_SOURCE_DIR "${PROJECT_SOURCE_DIR}/third_party/${NAME_LOWER}-src/")
      message(STATUS "No SOURCE_DIR provided for ${NAME}. Assuming: ${${NAME}_SOURCE_DIR}")
    endif()

    string(TOUPPER "${NAME}" NAME_UPPERCASE)

    # Find the library
    if(DEFINED arg_GIT_TAG)
      set(GIT_TAG_LINE GIT_TAG ${arg_GIT_TAG})
      if(arg_ALLOW_DIRTY)
        set(ALLOW_DIRTY_LINE ALLOW_DIRTY)
      endif()
    endif()
    gempic_confirm_git_tag(NAME ${NAME}
                LOCATION ${${NAME}_SOURCE_DIR}
                ${GIT_TAG_LINE}
                ${ALLOW_DIRTY_LINE})
    if(${${NAME}_CORRECT_VERSION_FOUND})
      set(GEMPICX_${NAME}_VERSION ${${NAME}_VERSION_FOUND} CACHE STRING
          "Version set by confirm_git_tag" FORCE)
      set(FETCHCONTENT_SOURCE_DIR_${NAME_UPPERCASE} ${${NAME}_SOURCE_DIR})

      # Strictly speaking unnecessary, because the library source was found
      set(GIT_REPOSITORY_LINE)
      set(GIT_TAG_LINE)
      set(GIT_UPDATE_LINE)
    else()
      unset(FETCHCONTENT_SOURCE_DIR_${NAME_UPPERCASE} CACHE)
      unset(FETCHCONTENT_SOURCE_DIR_${NAME_UPPERCASE})
      # Library wasn't found
      if(NOT DEFINED arg_GIT_REPOSITORY)
        message(FATAL_ERROR "gempic_FetchContent_Declare() requires a GIT_REPOSITORY when no "
                            "matching library is found. The call for ${NAME} gave none.")
      else()
        set(GIT_REPOSITORY_LINE GIT_REPOSITORY ${arg_GIT_REPOSITORY})
      endif()
      message(STATUS "No suitable source folders found. Fetching ${NAME} from ${arg_GIT_REPOSITORY} ...")
      # Ensure repository is updated (CHECKOUT) and changes are not permanently deleted (REBASE_)
      # See https://cmake.org/cmake/help/latest/module/ExternalProject.html#git
      #set(${NAME}_GIT_UPDATE_LINE GIT_REMOTE_UPDATE_STRATEGY CHECKOUT)
    endif()

    FetchContent_Declare(
      ${NAME}
      ${GIT_REPOSITORY_LINE}
      ${GIT_TAG_LINE}
      ${GIT_UPDATE_LINE}
      SOURCE_DIR ${${NAME}_SOURCE_DIR}
      ${arg_UNPARSED_ARGUMENTS}  # Give remaining arguments to FetchContent_Declare()
    )

    # Get version variables 
    if(NOT ${NAME}_CORRECT_VERSION_FOUND AND arg_GIT_TAG)
      set(GEMPICX_${NAME}_VERSION ${arg_GIT_TAG} CACHE STRING
          "Version downloaded by FetchContent" FORCE)
    endif()
    set(${NAME}_NOT_INSTALLED TRUE PARENT_SCOPE)
  endif()
  set(${NAME}_FOUND ${${NAME}_FOUND} PARENT_SCOPE)
endfunction()
