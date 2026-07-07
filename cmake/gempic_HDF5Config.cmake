include(${PROJECT_SOURCE_DIR}/cmake/gempic_FetchContent_Declare.cmake)

if(GEMPIC_USE_HDF5)
  option(GEMPIC_HDF5_PARALLEL "Enable parallel HDF5 (MPI)" ON)
  
  if(GEMPIC_HDF5_PARALLEL)
    set(HDF5_PREFER_PARALLEL TRUE)
  endif()
  if(HDF5_C_COMPILER_EXECUTABLE)
    set(HDF5_FIND_PACKAGE_ARGS COMPONENTS C)
  else()#if(HDF5_ROOT OR HDF5_DIR OR DEFINED ENV{HDF5_ROOT})
    set(HDF5_FIND_PACKAGE_ARGS CONFIG COMPONENTS C)
  endif()
  gempic_FetchContent_Declare(HDF5
    SOURCE_DIR     ${PROJECT_SOURCE_DIR}/third_party/hdf5-src
    GIT_REPOSITORY https://github.com/HDFGroup/hdf5.git
    GIT_TAG        2.1.1
    ALLOW_DIRTY    ${USE_DIRTY_HDF5_REPO}
    FIND_PACKAGE_ARGS ${HDF5_FIND_PACKAGE_ARGS}
  )
  if(NOT HDF5_FOUND)
    # ------------------------------------------------------------
    # Internal (FetchContent) HDF5 build
    # ------------------------------------------------------------
    set(HDF5_ENABLE_Z_LIB_SUPPORT OFF CACHE BOOL "" FORCE)
    set(HDF5_ENABLE_SZIP_ENCODING  OFF CACHE BOOL "" FORCE)
    set(HDF5_USE_LIBAEC            OFF CACHE BOOL "" FORCE)
    set(HDF5_ENABLE_SZIP_SUPPORT   OFF CACHE BOOL "" FORCE)
    set(HDF5_BUILD_CPP_LIB         OFF CACHE BOOL "" FORCE)
    set(HDF5_BUILD_FORTRAN         OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS          OFF CACHE BOOL "" FORCE)
        
    # Parallel HDF5 support (MPI)
    if(GEMPIC_HDF5_PARALLEL)
      set(HDF5_ENABLE_PARALLEL ON  CACHE BOOL "" FORCE)
    else()
      set(HDF5_ENABLE_PARALLEL OFF CACHE BOOL "" FORCE)
    endif()
    set(HDF5_BUILD_EXAMPLES        OFF CACHE BOOL "" FORCE)
    set(HDF5_BUILD_TOOLS           OFF CACHE BOOL "" FORCE)
    set(HDF5_BUILD_TESTING         OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(HDF5)
  endif()
  
  # ------------------------------------------------------------
  # Resolve backend target (name varies by HDF5 version/build)
  # ------------------------------------------------------------
  set(_hdf5_target "")
  if(TARGET HDF5::HDF5)
    set(_hdf5_target HDF5::HDF5)
  elseif(TARGET hdf5::hdf5)
    set(_hdf5_target hdf5::hdf5)
  elseif(TARGET hdf5-static)
    set(_hdf5_target hdf5-static)
  elseif(TARGET hdf5)
    set(_hdf5_target hdf5)
  endif()
  
  if(_hdf5_target STREQUAL "")
  message(FATAL_ERROR "HDF5 was configured but no usable target was found.")
  endif()
  
  # ------------------------------------------------------------
  # Canonical interface target - this is what everything links to
  # ------------------------------------------------------------
  if(NOT TARGET GEMPICX::HDF5)
    add_library(GEMPICX::HDF5 INTERFACE IMPORTED)
  endif()
  target_link_libraries(GEMPICX::HDF5 INTERFACE ${_hdf5_target})
    
  message(STATUS "HDF5 backend target: ${_hdf5_target}")
  message(STATUS "Canonical target: GEMPICX::HDF5")
  
  # ------------------------------------------------------------
  # Install - only needed when we built HDF5 ourselves
  # ------------------------------------------------------------
  if(NOT HDF5_FOUND)
    gempic_suppress_third_party_warnings(TARGET ${_hdf5_target})
    install(TARGETS ${_hdf5_target}
            EXPORT   GEMPICXTargets
            ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR})
    # HDF5 headers  hdf5_SOURCE_DIR is set by FetchContent
    install(DIRECTORY "${hdf5_SOURCE_DIR}/src/"
            DESTINATION     ${CMAKE_INSTALL_INCLUDEDIR}
            FILES_MATCHING  PATTERN "*.h")
  endif()
  # ------------------------------------------------------------
  # Cache variables for the config file to replay
  # ------------------------------------------------------------
  set(GEMPIC_HDF5_LINK_TARGET ${_hdf5_target} CACHE INTERNAL "")
  set(GEMPICX_HDF5_WAS_SYSTEM ${HDF5_FOUND}   CACHE INTERNAL "")
  
  if(NOT HDF5_FOUND)
    set(GEMPICX_HDF5_FETCHED_LIB_DIR "${CMAKE_INSTALL_LIBDIR}"     CACHE INTERNAL "")
    set(GEMPICX_HDF5_FETCHED_INC_DIR "${CMAKE_INSTALL_INCLUDEDIR}" CACHE INTERNAL "")
  endif()
endif()
