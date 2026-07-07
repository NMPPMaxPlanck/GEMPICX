function(gempic_enforce_separate_build_dir)
  file(REAL_PATH ${CMAKE_BINARY_DIR} REAL_PROJECT_BINARY_DIR)
  file(REAL_PATH ${CMAKE_CURRENT_SOURCE_DIR} REAL_PROJECT_SOURCE_DIR)
  if(${REAL_PROJECT_BINARY_DIR} STREQUAL ${REAL_PROJECT_SOURCE_DIR})
    message(FATAL_ERROR
            "GEMPIC: Building in source tree is not allowed.\n"
            "        Delete ${REAL_PROJECT_BINARY_DIR}/CMakeCache.txt and"
            " ${REAL_PROJECT_BINARY_DIR}/CMakeFiles.\n"
            "        Use `cmake -B /PATH/TO/BUILD/DIR ...` to set a different binary directory.")
  endif()
endfunction()

# For use in functions using cmake_parse_arguments
# Checks if variable is defined
# Must provide variable as argument to ARG
# Optionally provide a parse_arguments-added PREFIX to remove when printing  variable name
function(gempic_check_required_variable)
  set(oneValueArgs CHECK_VARIABLE PREFIX)
  cmake_parse_arguments("arg" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(NOT DEFINED arg_CHECK_VARIABLE)
    message(FATAL_ERROR "gempic_check_required_variable requires a non-empty CHECK_VARIABLE.")
  endif()
  if(NOT DEFINED ${arg_PREFIX}${arg_CHECK_VARIABLE} AND NOT DEFINED "${arg_PREFIX}_${arg_CHECK_VARIABLE}")
    message(FATAL_ERROR "The \"${arg_CHECK_VARIABLE}\" variable must be defined.")
  endif()
endfunction()

# For use in functions using cmake_parse_arguments
# Checks if variables are defined
# Must provide a list of variables as argument to ARGS
# Optionally provide a parse_arguments-added PREFIX to remove from ARGS name
function(gempic_check_required_variables)
  set(oneValueArgs PREFIX)
  set(multiValueArgs CHECK_VARIABLES)
  cmake_parse_arguments("arg" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(NOT DEFINED arg_CHECK_VARIABLES)
    message(FATAL_ERROR "gempic_check_required_variable requires a non-empty CHECK_VARIABLES.")
  endif()
  foreach(SINGLE_ARG ${arg_CHECK_VARIABLES})
    if(DEFINED arg_PREFIX)
      gempic_check_required_variable(CHECK_VARIABLE ${SINGLE_ARG} PREFIX "${arg_PREFIX}")
    else()
      gempic_check_required_variable(CHECK_VARIABLE ${SINGLE_ARG})
    endif()
  endforeach()
endfunction()

function(gempic_suppress_third_party_warnings)
  set(mandatoryArgs TARGET)
  set(oneValueArgs ${mandatoryArgs})
  cmake_parse_arguments("arg" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  gempic_check_required_variables(CHECK_VARIABLES ${mandatoryArgs} PREFIX arg)

  # Don't attempt to modify installed targets
  if(${arg_TARGET}_FOUND)
    return()
  endif()

  # Suppress upstream warnings
  get_target_property(${arg_TARGET}_HEADER_DIRS ${arg_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
  
  # Suppress warnings about source-prefixed include paths
  get_target_property(${arg_TARGET}_ISID ${arg_TARGET} INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
  if(${arg_TARGET}_ISID MATCHES "BUILD_INTERFACE")
    set(_${arg_TARGET}_ISID "$<BUILD_INTERFACE:${${arg_TARGET}_HEADER_DIRS}>" "${${arg_TARGET}_ISID}")
  elseif(${arg_TARGET}_ISID)
    set(_${arg_TARGET}_ISID "$<BUILD_INTERFACE:${${arg_TARGET}_HEADER_DIRS}" "${${arg_TARGET}_ISID}>")
  else()
    set(_${arg_TARGET}_ISID "$<BUILD_INTERFACE:${${arg_TARGET}_HEADER_DIRS}>")
  endif()

  set_target_properties(${arg_TARGET} PROPERTIES
      INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
      "${_${arg_TARGET}_ISID}")
  # Using the big guns
  if(MSVC)
    target_compile_options(${arg_TARGET} PRIVATE /W0)
  else()
    target_compile_options(${arg_TARGET} PRIVATE -w)
  endif()
endfunction()
