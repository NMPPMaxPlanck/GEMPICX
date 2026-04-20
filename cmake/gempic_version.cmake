# Retrieves GEMPICX version info and sets internal cache variables
# GEMPICX_GIT_VERSION, GEMPICX_PKG_VERSION, and GEMPICX_RELEASE_NUMBER

include(cmake/gempic_utils.cmake)

# Helper functions
# Pads a string INPUT with PADCHAR until it is PADLENGTH long, then returns it as OUTPUT
# INPUT may be empty
function(gempic_pad_string)
  set(mandatoryArgs OUTPUT PADCHAR PADLENGTH)
  set(oneValueArgs INPUT ${mandatoryArgs})
  cmake_parse_arguments("arg" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  gempic_check_required_variables(CHECK_VARIABLES ${mandatoryArgs} PREFIX arg)
  string(LENGTH "${arg_INPUT}" strlen)
  math(EXPR strlen "${arg_PADLENGTH} - ${strlen}")

  if(strlen GREATER 0)
    string(REPEAT ${arg_PADCHAR} ${strlen} _pad)
    string(PREPEND arg_INPUT ${_pad})
  endif()

  set(${arg_OUTPUT} "${arg_INPUT}" PARENT_SCOPE)
endfunction()

# Pads an entry INDEX of list LIST using the gempic_pad_string function
# If INDEX is bigger than the length of the list, pads ""
function(gempic_pad_list_entry)
  set(oneValueArgs OUTPUT INDEX PADCHAR PADLENGTH)
  set(multiValueArgs LIST)
  cmake_parse_arguments("arg" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  # gempic_check_required_variable(CHECK_VARIABLE OUTPUT PREFIX arg)
  gempic_check_required_variables(CHECK_VARIABLES ${oneValueArgs} ${multiValueArgs} PREFIX arg)
  list(LENGTH arg_LIST listlen)
  if(listlen GREATER arg_INDEX)
    list(GET arg_LIST ${arg_INDEX} input)
    gempic_pad_string(
      OUTPUT pad_string_out
      INPUT ${input}
      PADCHAR ${arg_PADCHAR}
      PADLENGTH ${arg_PADLENGTH})
  else()
    gempic_pad_string(
      OUTPUT pad_string_out
      PADCHAR ${arg_PADCHAR}
      PADLENGTH ${arg_PADLENGTH})
  endif()
  set(${arg_OUTPUT} "${pad_string_out}" PARENT_SCOPE)
endfunction()

function(gempic_create_and_link_cmake_constants_file)
  set(PRECONFIG_INPUT "${PROJECT_SOURCE_DIR}/cmake/gempic_CMakeConstants.H.in")
  configure_file(
    "${PRECONFIG_INPUT}"
    "${PROJECT_SOURCE_DIR}/Src/Utils/GEMPIC_CMakeConstants.H"
  )
endfunction()

# Retrieves GEMPICX version info via git and sets internal cache variables
# GEMPICX_GIT_VERSION, GEMPICX_PKG_VERSION, and GEMPICX_RELEASE_NUMBER
function(gempic_get_version)
  find_package(Git)

  set(_tmp "")

  # Try to inquire software version from git
  # GEMPICX_GIT_VERSION is a string of the form
  # v<tag>[-<#commits since tag>]-<short SHA>[-dirty]
  # Where
  # <tag> is the latest tag
  # <#commits since tag> only exists if commit is not directly on a tag
  # <short SHA> is the shortened commit ID
  # -dirty appears if the repository contains uncommitted changes
  if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/.git AND ${GIT_FOUND})
    execute_process(COMMAND git merge-base HEAD master
                    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    OUTPUT_VARIABLE _master_commit
                    ERROR_QUIET)
    execute_process(COMMAND git describe ${_master_commit} --always --tags
                    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    OUTPUT_VARIABLE _tmp
                    ERROR_QUIET)
    # filter invalid descriptions in shallow git clones
    if(NOT _tmp MATCHES "^v([0-9]+)\\.([0-9]+)(\\.([0-9]+))*(-.*)*$")
      set(_tmp "")
    endif()
  endif()

  set(GEMPICX_GIT_VERSION "${_tmp}" CACHE INTERNAL "" FORCE)

  # GEMPICX_PKG_VERSIOIN
  # Package version is a modified form of GEMPICX_GIT_VERSION:
  # Major version is the major version of the tag
  # Minor version is the minor version of the tag
  # Patch version is the number of commits since the tag
  # Tweak version is a boolean indicating whether or not uncommitted changes are in the repository
  if(GEMPICX_GIT_VERSION)
    # Remove "v" from the beginning
    string(SUBSTRING "${GEMPICX_GIT_VERSION}" 1 -1 _pkg_version)
    # Remove shortened SHA of commit
    string(FIND "${_pkg_version}" "-" _idx REVERSE)
    string(SUBSTRING "${_pkg_version}" 0 "${_idx}" _pkg_version)
    if(_pkg_version MATCHES "-")
      # Number of commits since tag exists and is patch version number 
      string(REPLACE "-" "." _pkg_version "${_pkg_version}")
      set(_pkg_version "${_pkg_version}")
    else()
      # We're directly on a tag and add '0' as patch version number 
      set(_pkg_version "${_pkg_version}.0")
    endif()
  endif()

  set(GEMPICX_PKG_VERSION "${_pkg_version}" CACHE INTERNAL "" FORCE)

  # GEMPICX_RELEASE_NUMBER
  # Release number is a zero-padded integer variant of package version
  # Padding is arbitrary, but hopefully long enough that all release numbers have the same length
  if(GEMPICX_GIT_VERSION)
    string(REPLACE "." ";" _version_list ${_pkg_version})
    gempic_pad_list_entry(
                   OUTPUT _major_version
                   LIST "${_version_list}"
                   INDEX 0
                   PADCHAR "0"
                   PADLENGTH 2)
    gempic_pad_list_entry(
                   OUTPUT _minor_version
                   LIST "${_version_list}"
                   INDEX 1
                   PADCHAR "0"
                   PADLENGTH 2)
    gempic_pad_list_entry(
                   OUTPUT _patch_version
                   LIST "${_version_list}"
                   INDEX 2
                   PADCHAR "0"
                   PADLENGTH 4)
    string(CONCAT _rel_number "${_major_version}" "${_minor_version}" "${_patch_version}" "${_tweak_version}")
  endif()

  set(GEMPICX_RELEASE_NUMBER "${_rel_number}" CACHE INTERNAL "" FORCE)

  if(NOT GEMPICX_GIT_VERSION)
    set(GEMPICX_GIT_VERSION "Unknown (not a git repository)" CACHE INTERNAL "" FORCE)
  endif()
endfunction()