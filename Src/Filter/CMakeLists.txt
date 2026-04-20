if (AMReX_FFT)
  set(FFTSource GEMPIC_FFTFilter.cpp)
endif()

target_sources(GEMPICX PRIVATE
  GEMPIC_Filter.cpp
  ${FFTSource})
