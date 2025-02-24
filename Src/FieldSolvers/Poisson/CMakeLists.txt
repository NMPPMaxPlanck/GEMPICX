if (AMReX_FFT)
  set(FFTFile GEMPIC_FFT.cpp)
endif()
target_sources(GEMPICX PRIVATE GEMPIC_PoissonSolver.cpp ${FFTFile})
