# Writes the build directory into a file read by clang-tidy
# (or in case you forget where you built GEMPIC to)
file(WRITE ${CMAKE_SOURCE_DIR}/scripts/build_dir.txt "${CMAKE_BINARY_DIR}")

# If the project was not built with clang, clang(-tidy) needs implicit includes
#if(CMAKE_CXX_COMPILER_ID MATCHES "GNU.*")
  # but specifically not .../x86_64-linux-gnu/<VERSION>/include
  string(REGEX REPLACE ";[^;]*gnu/[0-9\.]*/include;" ";" SYS_INCLUDE_STRING "\n;${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES}")
  string(REPLACE ";" " --extra-arg=-isystem --extra-arg=" SYS_INCLUDE_STRING "${SYS_INCLUDE_STRING}")
  set(CLANG_TIDY_ARG_STRING "${SYS_INCLUDE_STRING} --extra-arg=-std=c++${CMAKE_CXX_STANDARD}")
#endif()
file(APPEND ${CMAKE_SOURCE_DIR}/scripts/build_dir.txt "${CLANG_TIDY_ARG_STRING}")