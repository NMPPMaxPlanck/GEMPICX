if (AMReX_FFT)
  set(FFTSource GEMPIC_PoissonFFT.cpp)
endif()
target_sources(GEMPICX PRIVATE
               GEMPIC_PoissonSolver.cpp
               ${FFTSource})