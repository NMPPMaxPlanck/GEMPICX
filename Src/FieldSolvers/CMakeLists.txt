# Add a library to this build. The name of the library is GEM_FIELD and it
# consists of only the gem_maxwell.cpp file
#add_library(GEM_FIELD gem_maxwell_yee.cpp)

add_subdirectory(Poisson)
add_subdirectory(Quasineutral)
add_subdirectory(Testing)
