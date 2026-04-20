# Function for starting a collapsible section
start_details () {
  details_id="${1}"
  details_title="${2:-$details_id}"

  echo "section_start:`date +%s`:${details_id}[collapsed=true]"$'\r\e[0K'"${details_title}"
}

# Function for ending the collapsible section
end_details () {
  details_id="${1}"

  echo "section_end:`date +%s`:${details_id}"$'\r\e[0K'
}

configure_build () {
  configure_flags="${1}"
  numprocs="${2:-$(nproc)}"
  build_dir="${3:-build}"

  start_details configure "CMake Configure Step"
  cmake -S . -B $build_dir $configure_flags
  end_details configure
  start_details build "CMake Build Step"
  bash -c 'cmake --build "$1" -j "$2" > >(tee build.out) 2> >(tee build.err >&2)' _ "$build_dir" "$numprocs"
  end_details build
}

run_ctest () {
  numprocs="${1:-$(nproc)}"
  build_dir="${2:-build}"
  ctest_args="${3}"
  start_details test "Test Step"
  ctest --output-on-failure --no-tests=error -j $numprocs --test-dir $build_dir $ctest_args
  end_details test
}

install_test () {
  cxx_compiler="${1:-mpicxx}"
  c_compiler="${2:-mpicc}"
  build_dir="${3:-build}"
  relative_install_dir="${4:-install}"
  test_flags="${5}"

  install_test_dir=Examples/BuildGEMPICXInstalled
  install_test_build_dir=$install_test_dir/$build_dir

  start_details install_test
  start_details install
  cmake --install $build_dir --prefix=$relative_install_dir
  end_details install
  start_details configure_build_install_test
  cmake -S$install_test_dir -B$install_test_build_dir -DGEMPICX_ROOT=$(pwd)/$relative_install_dir -DCMAKE_CXX_COMPILER=$cxx_compiler -DCMAKE_C_COMPILER=$c_compiler $test_flags
  cmake --build $install_test_build_dir
  end_details configure_build_install_test
  ctest --output-on-failure --no-tests=error --test-dir $install_test_build_dir
  end_details install_test
}

# convenience helper for gpu
install_test_gpu () {
  cxx_compiler="$1"
  c_compiler="$2"
  build_dir="$3"
  install_dir="$4"
  test_flags="$5 -DUSE_CUDA=ON"

  install_test "$cxx_compiler" "$c_compiler" "$build_dir" "$install_dir" "$test_flags"
}