if (AMReX_FFT)
  set(VlasovMaxwellSemiImplicitFFTFile GEMPIC_VlasovMaxwellSemiImplicitFFT.cpp)
endif()
target_sources(GEMPICX PRIVATE ${VlasovMaxwellSemiImplicitFFTFile})
