if (AMReX_FFT)
  set(FFTFile GEMPIC_FFTFilter.cpp)
endif()

target_sources(GEMPICX PRIVATE GEMPIC_Filter.cpp ${FFTFile})
