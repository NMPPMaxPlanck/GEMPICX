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

# Function for configure/build/test jobs
configure_build_test () {
  preset="${1}"
  build_dir="${2:-build}"
  numprocs="${3:-$(nproc)}"

  start_details configure "CMake Configure Step"
  cmake -S . -B $build_dir --preset $preset
  end_details configure
  start_details build "CMake Build Step"
  cmake --build $build_dir -j $numprocs
  end_details build
  start_details test "Test Step"
  cd $build_dir
  ctest --output-on-failure --no-tests=error -j $numprocs
  end_details test
}

configure_build_test_gpu () {
  preset="${1}"
  build_dir="${2:-build}"
  numprocs="${3:-$(nproc)}"

  start_details configure "CMake Configure Step"
  cmake -S . -B $build_dir --preset $preset
  end_details configure
  start_details build "CMake Build Step"
  cmake --build $build_dir -j $numprocs
  end_details build
  start_details test "Test Step"
  cd $build_dir
  ctest --output-on-failure --no-tests=error
  end_details test
}