# Documentation can be built stand-alone
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.14.0)
  project("gempic-doc" LANGUAGES NONE)
  include(../cmake/gempic_version.cmake)
endif()

add_subdirectory(doxygen)
add_subdirectory(sphinx)
