# AMReX Copyright (c) 2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S. Dept. of Energy).  All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# (1) Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 
# (2) Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# 
# (3) Neither the name of the University of California, Lawrence Berkeley
# National Laboratory, U.S. Dept. of Energy nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
# 
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Checks for FFT capabilities without crashing and burning if they're not found.
# Stolen from AMReX (Tools/CMake/FindAMReXFFTW.cmake),
# who reasonably have no way to check for FFT without erroring out upon failure.
if(GEMPIC_USE_CUDA)
  # If cuda, we use cufft. If you have CUDA and a religious aversion to cufft, I deeply apologise.
  set(AMReX_FFT ON CACHE BOOL "AMReX Option set within GEMPIC")
elseif(GEMPIC_USE_HIP)
  # If ROCm, we use rocFFT and also shouldn't check.
  set(AMReX_FFT ON CACHE BOOL "AMReX Option set within GEMPIC")
else()
  # Central FFTW3 Search ###############################################
  #
  # On Windows, try searching for FFTW3(f)Config.cmake files first
  #   Installed .pc files wrongly and unconditionally add -lm
  #   https://github.com/FFTW/fftw3/issues/236

  # On Linux & macOS, note Autotools install bug:
  #   https://github.com/FFTW/fftw3/issues/235
  # Thus, rely on .pc files

  set(AMReX_FFTW_SEARCH_VALUES PKGCONFIG CMAKE)
  set(AMReX_FFTW_SEARCH_DEFAULT PKGCONFIG)
  if(WIN32)
      set(AMReX_FFTW_SEARCH_DEFAULT CMAKE)
  endif()
  set(AMReX_FFTW_SEARCH ${AMReX_FFTW_SEARCH_DEFAULT}
          CACHE STRING "FFTW search method (PKGCONFIG/CMAKE)")
  set_property(CACHE AMReX_FFTW_SEARCH PROPERTY STRINGS ${AMReX_FFTW_SEARCH_VALUES})
  if(NOT AMReX_FFTW_SEARCH IN_LIST AMReX_FFTW_SEARCH_VALUES)
      message(FATAL_ERROR "AMReX_FFTW_SEARCH (${AMReX_FFTW_SEARCH}) must be one of ${AMReX_FFTW_SEARCH_VALUES}")
  endif()
  mark_as_advanced(AMReX_FFTW_SEARCH)

  function(fftw_quiet_find_precision HFFTWp)
    if(AMReX_FFTW_SEARCH STREQUAL CMAKE)
      find_package(FFTW3${HFFTWp} CONFIG QUIET)
    else()
      find_package(PkgConfig QUIET)
      if (PKG_CONFIG_FOUND)
        pkg_check_modules(fftw3${HFFTWp} IMPORTED_TARGET fftw3${HFFTWp})
      endif()
    endif()
  endfunction()

  # floating point precision suffixes: we request float and double precision
  fftw_quiet_find_precision("")
  if(FFTW3_FOUND OR fftw3_FOUND)
    fftw_quiet_find_precision("f")
    if(FFTW3f_FOUND OR fftw3f_FOUND)
      set(AMReX_FFT ON CACHE BOOL "AMReX Option set within GEMPIC")
    else()
      message(STATUS "Failed to find fftw3 float precision. No FFT will be available.")
      set(AMReX_FFT OFF CACHE BOOL "AMReX Option set within GEMPIC")
    endif()
  else()
    message(STATUS "Failed to find fftw3 double precision. No FFT will be available.")
    set(AMReX_FFT OFF CACHE BOOL "AMReX Option set within GEMPIC")
  endif()
endif()