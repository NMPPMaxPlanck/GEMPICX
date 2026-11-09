if (AMReX_HYPRE)
  set(HypreDebugFile GEMPIC_DKQuasineutralDebug.cpp)
  target_sources(GEMPICX PRIVATE ${HypreDebugFile} GEMPIC_DKQuasineutralImpl.cpp)
endif()

