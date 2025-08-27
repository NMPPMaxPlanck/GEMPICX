if (AMReX_HYPRE)
  set(HypreDebugFile GEMPIC_QuasineutralDebug.cpp)
  target_sources(GEMPICX PRIVATE ${HypreDebugFile})
endif()

