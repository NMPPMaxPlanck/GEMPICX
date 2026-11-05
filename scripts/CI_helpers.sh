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

# Number of CPUs the job may use: the CFS quota of the cgroup (docker --cpus) if smaller than nproc.
# nproc only sees the cpuset, e.g. 8 on the NVIDIA runners, which have a quota of 4 CPUs.
available_cpus () {
  local n quota period
  n=$(nproc)
  if { read -r quota period < /sys/fs/cgroup/cpu.max; } 2>/dev/null && [[ $quota =~ ^[0-9]+$ ]]; then
    quota=$(( (quota + period - 1) / period ))
    (( quota < n )) && n=$quota
  fi
  echo $n
}

configure_build () {
  configure_flags="${1}"
  numprocs="${2:-$(available_cpus)}"
  build_dir="${3:-build}"

  start_details configure "CMake Configure Step"
  cmake -S . -B $build_dir $configure_flags
  end_details configure
  start_details build "CMake Build Step"
  bash -c 'cmake --build "$1" -j "$2" > >(tee build.out) 2> >(tee build.err >&2)' _ "$build_dir" "$numprocs"
  end_details build
}

# Reuse hypre, Kokkos and AMReX builds across pipelines via the project's package registry (one
# generic package per configuration, e.g. cuda-3D). The dependencies are configured with the job's
# flags, and the version is a hash of that configured build (scripts/CI_dependencies_key.py), so it
# changes whenever the dependencies would be built or installed differently. On a miss, they are
# built and uploaded. Afterwards CMAKE_PREFIX_PATH points to the install, where gempic's
# find_package() finds it. Like the other helpers, it relies on the CI shell stopping at the first
# error.
prebuilt_dependencies () {
  dependencies_name="${1}"
  configure_flags="${2}"
  dependencies_build_dir=build-dependencies
  dependencies_install_dir=$(pwd)/dependencies-install

  start_details dependencies "Prebuilt dependencies"
  # Ninja, for its per-directory targets: only the dependencies are built (_deps/<name>-build/all)
  CMAKE_GENERATOR=Ninja cmake -S . -B $dependencies_build_dir $configure_flags \
    -DGEMPIC_BUILD_TESTS=OFF -DGEMPIC_BUILD_EXAMPLES=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  dependencies_key=$( { echo "${CI_JOB_IMAGE}"; module list 2>&1; declare -f prebuilt_dependencies; } |
                      python3 scripts/CI_dependencies_key.py $dependencies_build_dir )
  dependencies_url="${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/packages/generic/${dependencies_name}/${dependencies_key}/dependencies.tar.gz"
  echo "Dependencies ${dependencies_name}, key ${dependencies_key}"
  if curl --fail --silent --show-error --header "JOB-TOKEN: ${CI_JOB_TOKEN}" \
       --output dependencies.tar.gz "${dependencies_url}"; then
    tar -xzf dependencies.tar.gz
  else
    echo "No prebuilt dependencies for this key, building them"
    # hypre only exists if AMReX uses it (CPU builds)
    dependencies=$(cd $dependencies_build_dir/_deps &&
                   ls -d hypre-build kokkos-build amrex-build 2>/dev/null || true)
    cmake --build $dependencies_build_dir -j $(available_cpus) --target $(printf '_deps/%s/all ' $dependencies)
    for dependency in $dependencies; do
      cmake --install $dependencies_build_dir/_deps/$dependency --prefix $dependencies_install_dir
    done
    tar -czf dependencies.tar.gz dependencies-install
    curl --fail --silent --show-error --header "JOB-TOKEN: ${CI_JOB_TOKEN}" \
      --upload-file dependencies.tar.gz "${dependencies_url}" || echo "Upload of the dependencies failed"
  fi
  rm -rf $dependencies_build_dir dependencies.tar.gz
  export CMAKE_PREFIX_PATH="${dependencies_install_dir}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
  end_details dependencies
}

run_ctest () {
  numprocs="${1:-$(available_cpus)}"
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
  test_flags=("${@:5}")  # capture all remaining args as array elements

  install_test_dir=Examples/BuildGEMPICXInstalled
  install_test_build_dir=$install_test_dir/$build_dir

  start_details install_test
  start_details install
  cmake --install $build_dir --prefix=$relative_install_dir
  end_details install
  start_details configure_build_install_test
  cmake -S$install_test_dir -B$install_test_build_dir -DGEMPICX_ROOT=$(pwd)/$relative_install_dir -DCMAKE_CXX_COMPILER=$cxx_compiler -DCMAKE_C_COMPILER=$c_compiler "${test_flags[@]}"
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
  test_flags=("${@:5}" "-DUSE_CUDA=ON")

  install_test "$cxx_compiler" "$c_compiler" "$build_dir" "$install_dir" "${test_flags[@]}"
}
