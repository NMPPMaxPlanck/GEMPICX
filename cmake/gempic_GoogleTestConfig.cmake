include(${PROJECT_SOURCE_DIR}/cmake/gempic_FetchContent_Declare.cmake)

gempic_FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
  SOURCE_DIR     ${PROJECT_SOURCE_DIR}/third_party/googletest-src
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

foreach(target gtest gmock gtest_main gmock_main)
  if(NOT TARGET ${target})
    continue()
  endif()
  gempic_suppress_third_party_warnings(TARGET ${target})
endforeach()
