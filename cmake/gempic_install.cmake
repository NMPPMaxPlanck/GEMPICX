# Contains helper functions for installation

macro(gempic_set_default_install_directories)
  include(GNUInstallDirs) 
  # Provide a default install directory
  if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
      set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/install"
          CACHE PATH "GEMPICX installation directory" FORCE)
      # Stop Hypre from redefining this default:
      unset(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT CACHE) # Unsets cache variable
      unset(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT) # Unsets local variable
    endif()
  endif()
  # Base path configurable with CMAKE_INSTALL_PREFIX
  set(GEMPICX_BINDIR ${CMAKE_INSTALL_BINDIR})
  set(GEMPICX_LIBDIR ${CMAKE_INSTALL_LIBDIR})
  set(GEMPICX_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR})
endmacro()

# Collect all currently added targets in all subdirectories except third_party
# Stolen from ilya1725
# https://stackoverflow.com/questions/60211516/programmatically-get-all-targets-in-a-cmake-project
# Parameters:
# - _result the list containing all found targets
# - _dir root directory to start looking from
function(get_all_targets _result _dir)
  get_property(_subdirs DIRECTORY "${_dir}" PROPERTY SUBDIRECTORIES)
  foreach(_subdir IN LISTS _subdirs)
    if(NOT "${_subdir}" MATCHES "third_party")
      get_all_targets(${_result} "${_subdir}")
    endif()
  endforeach()

  get_directory_property(_sub_targets DIRECTORY "${_dir}" BUILDSYSTEM_TARGETS)
  set(${_result} ${${_result}} ${_sub_targets} PARENT_SCOPE)
endfunction()

# Remove all  subdirectories from a list, leaving only the shortest paths
# Example:
# /b/;/a/b/c/;/a/  becomes  /b/;/a/
# because /a/b/c/ is a subdirectory of /a/
function(remove_subdirs _result)
  foreach(testdir ${${_result}})
    foreach(otherdir ${${_result}})
      if(NOT testdir STREQUAL otherdir)
        # test if testdir is a subdirectory of otherdir
        cmake_path(IS_PREFIX otherdir ${testdir} NORMALIZE is_subdir)
        # if yes, add it to the kill list
        if(is_subdir)
          list(APPEND TOO_REMOVE ${testdir})
          break()
        endif()
      endif()
    endforeach()
  endforeach()
  # remove all paths on the kill list
  list(REMOVE_ITEM ${_result} ${TOO_REMOVE})
  set(${_result} ${${_result}} PARENT_SCOPE)
endfunction()

# Installs all defined targets into CMAKE_INSTALL_PREFIX directory.
# If not yet installed packages were provided via FetchContent, installs those as well.
# Ensures that users can link to individual files without providing entire/relative file path, i.e.
# #include "GEMPIC_Version.H"
#  not
# #include "Src/Utils/GEMPIC_Version.H"
macro(gempic_install_export_targets)
  get_all_targets(GEMPICX_TARGETS ${PROJECT_SOURCE_DIR})
  list(APPEND GEMPICX_TARGETS cmake_git_version_tracking) # Also needed, and I don't know why
  # https://discourse.cmake.org/t/why-do-private-linked-targets-leak-out-in-install-targets/13076/8
  foreach(_target ${GEMPICX_TARGETS})
    # List of all (public/interface) header directories of _target
    get_target_property(${_target}_HEADER_DIRS ${_target} INTERFACE_INCLUDE_DIRECTORIES)
    if(NOT ${_target}_HEADER_DIRS STREQUAL "${_target}_HEADER_DIRS-NOTFOUND")
      # Not allowing generator expressions makes this much simpler
      # Hopefully also means install-related decisions are localised to this file
      if(${_target}_HEADER_DIRS MATCHES "BUILD_INTERFACE" OR
         ${_target}_HEADER_DIRS MATCHES "INSTALL_INTERFACE")
        message(FATAL_ERROR "Generator expressions in link statements not supported for install\n"
                            "TARGET ${_target}\n"
                            "INTERFACE_INCLUDE_DIRECTORIES: ${${_target}_HEADER_DIRS}")
      else()
        # Convert all build header paths (whose prefix is this repo) to install paths
        # Ensures #include to individual files without providing entire/relative file path
        foreach(_target_header ${${_target}_HEADER_DIRS})
          file(RELATIVE_PATH ${_target}_header_include_path ${PROJECT_SOURCE_DIR} ${_target_header})
          list(APPEND ${_target}_INCLUDE_DIRS
                      "${GEMPICX_INCLUDEDIR}/${${_target}_header_include_path}")
        endforeach()
        
        set_property(TARGET ${_target} PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                    "$<BUILD_INTERFACE:${${_target}_HEADER_DIRS}>"
                    "$<INSTALL_INTERFACE:${${_target}_INCLUDE_DIRS}>")
        # Store the location of all include directories
        list(APPEND GEMPICX_TARGETS_HEADER_DIRS ${${_target}_HEADER_DIRS})
      endif()
    endif()
  endforeach()
  
  # Trim and use the include directories list for the install
  # It's also possible to avoid this step with HEADER FILE_SETs, but this requires:
  # 1. Adding (writing out explicitly) all header files to the individual targets
  # 2. Explicitly mentioning all the FILE_SETs in the defaults install command below
  list(REMOVE_DUPLICATES GEMPICX_TARGETS_HEADER_DIRS)
  remove_subdirs(GEMPICX_TARGETS_HEADER_DIRS)
  message(STATUS "All include dirs: ${GEMPICX_TARGETS_HEADER_DIRS}")
  install(DIRECTORY ${GEMPICX_TARGETS_HEADER_DIRS}
          DESTINATION ${GEMPICX_INCLUDEDIR}
          FILES_MATCHING PATTERN "*.H")

  # Set defaults
  install(
    TARGETS ${GEMPICX_TARGETS}
    EXPORT GEMPICXTargets
    RUNTIME DESTINATION ${GEMPICX_BINDIR}
    ARCHIVE DESTINATION ${GEMPICX_LIBDIR}
    LIBRARY DESTINATION ${GEMPICX_LIBDIR}
    INCLUDES DESTINATION ${GEMPICX_INCLUDEDIR}
    PUBLIC_HEADER DESTINATION ${GEMPICX_INCLUDEDIR}
    )

  # We must take responsibility for FetchContent'ed modules
  if(${HYPRE_NOT_INSTALLED})
    set(HYPRE_INSTALL_DIR ${GEMPICX_LIBDIR}/cmake/HYPRE
        CACHE PATH "Set by GEMPICX install" FORCE)
  endif()
  
  install(
    EXPORT GEMPICXTargets
    NAMESPACE GEMPICX::
    DESTINATION ${GEMPICX_LIBDIR}/cmake/GEMPICX)

  include(CMakePackageConfigHelpers)

  configure_package_config_file(${PROJECT_SOURCE_DIR}/cmake/GEMPICXConfig.cmake.in
    ${PROJECT_BINARY_DIR}/GEMPICXConfig.cmake
    INSTALL_DESTINATION ${GEMPICX_LIBDIR}/cmake/GEMPICX)

  if(GEMPICX_PKG_VERSION)
    set(GEMPICX_VERSION_FILE ${PROJECT_BINARY_DIR}/GEMPICXConfigVersion.cmake)
    write_basic_package_version_file(
      ${GEMPICX_VERSION_FILE}
      COMPATIBILITY ExactVersion)
  endif()

  install(FILES
    ${PROJECT_BINARY_DIR}/GEMPICXConfig.cmake
    ${GEMPICX_VERSION_FILE}
    DESTINATION ${GEMPICX_LIBDIR}/cmake/GEMPICX)
endmacro()
