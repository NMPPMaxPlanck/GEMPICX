include(${PROJECT_SOURCE_DIR}/cmake/gempic_FetchContent_Declare.cmake)

if(GEMPIC_USE_HDF5)
  set(HDF5_PREFER_PARALLEL ${AMReX_MPI})
  gempic_FetchContent_Declare(HDF5
    SOURCE_DIR     ${PROJECT_SOURCE_DIR}/third_party/hdf5-src
    GIT_REPOSITORY https://github.com/HDFGroup/hdf5.git
    GIT_TAG        2.1.1
    ALLOW_DIRTY    ${USE_DIRTY_HDF5_REPO}
    FIND_PACKAGE_ARGS COMPONENTS C
  )
  if(NOT HDF5_FOUND)
    if(AMReX_MPI)
      set(HDF5_ENABLE_PARALLEL ON  CACHE BOOL "HDF5 Option set within GEMPIC")
    else()
      set(HDF5_ENABLE_PARALLEL OFF CACHE BOOL "HDF5 Option set within GEMPIC")
    endif()
    set(HDF5_BUILD_EXAMPLES    OFF CACHE BOOL "HDF5 Option set within GEMPIC")
    set(HDF5_BUILD_HL_LIB      OFF CACHE BOOL "HDF5 Option set within GEMPIC")
    set(HDF5_BUILD_TOOLS       OFF CACHE BOOL "HDF5 Option set within GEMPIC")
    set(_crq_org ${CMAKE_REQUIRED_QUIET}) # Avoid check file spam from HDF5
    set(CMAKE_REQUIRED_QUIET ON)

    FetchContent_MakeAvailable(HDF5)
    set(CMAKE_REQUIRED_QUIET ${_crq_org})
  endif()

  if(NOT HDF5_FOUND)
    gempic_suppress_third_party_warnings(TARGET hdf5-static)
    add_library(hdf5::hdf5 ALIAS hdf5-static)
    install(TARGETS hdf5-static
            EXPORT   GEMPICXTargets
            ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR})
    # HDF5 headers  hdf5_SOURCE_DIR is set by FetchContent
    install(DIRECTORY "${hdf5_SOURCE_DIR}/src/"
            DESTINATION     ${CMAKE_INSTALL_INCLUDEDIR}
            FILES_MATCHING  PATTERN "*.h")
  endif()
endif()
