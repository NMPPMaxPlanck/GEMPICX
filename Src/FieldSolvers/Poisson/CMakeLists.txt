if (AMReX_FFT)
  set(FFTSource GEMPIC_FFT.cpp)
endif()
target_sources(GEMPICX PRIVATE
               GEMPIC_PoissonSolver.cpp
               ${FFTSource})