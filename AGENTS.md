# AGENTS.md

GEMPICX: C++20 plasma simulation framework (geometric electromagnetic particle-in-cell) on AMReX
and Kokkos, running on CPU (MPI) and GPU (CUDA/HIP). The dimension is fixed at compile time
(`AMReX_SPACEDIM` = 1, 2 or 3), so much of the code is dimension-dependent. Human documentation:
`README.md`, `doc/`, https://gempic.pages.mpcdf.de/gempic/.

## Layout

- `Src/<Module>/` library `GEMPICX`; `Examples/<Name>/` simulation drivers with input files.
- `Testing/<Module>/` GoogleTest tests, all built into one executable `GtestTests`.
- `third_party/` fetched dependencies (AMReX, Kokkos, hypre, googletest). **Never edit.**

## Build and test

```sh
cmake --preset cpu-release-3D   # configures build/<preset>; cmake --list-presets for others
cmake --build build/cpu-release-3D -j <N> [--target GEMPICX|GtestTests|<example>]
ctest --test-dir build/cpu-release-3D --output-on-failure -j <N>
cd build/cpu-release-3D/Testing && ./GtestTests --gtest_filter='FDDeRhamComplexTest.*'
```

- Full builds are slow (mostly the tests): build only the target you need, or configure with
  `-DGEMPIC_BUILD_TESTS=OFF` / `-DGEMPIC_BUILD_EXAMPLES=OFF`.
- One build directory per preset. Machine-specific presets go in `CMakeUserPresets.json`.
- CUDA builds use Kokkos `nvcc_wrapper` as the C++ compiler. Run ctest with `-j 1` on GPU builds.
- CI tests 1D, 2D and 3D: dimension-dependent changes must pass in all three.
- New sources and tests must be added to `target_sources(...)` in their folder's
  `CMakeLists.txt` (tests are named `<Name>_test.cpp`).

## Code style

Run `python3 scripts/sanitize_with_python.py` (clang-format + clang-tidy; `-format` for
clang-format only) after staging. It uses the most recently configured build
(`scripts/build_dir.txt`), so configure a CPU 3D build last: clang-tidy cannot use the compile
commands of a GPU build, and 3D-only code is only checked in 3D.
CI uses clang-format/clang-tidy 18; other versions can produce a different diff. The
merge-request CI job fails on any diff it produces, and on new compiler warnings. Full rules:
`doc/latex/CodeStyle.tex`.

- Files `GEMPIC_CamelCase.H/.cpp`; classes `CamelCase`; functions `snake_case`; variables
  `camelBack` (`E`, `Bx` allowed); members `m_`, static members `s_`.
- Namespaces follow folders (`Gempic::<Module>`, internals in `::Impl`); no `using namespace` in
  headers.
- Errors via `GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE` / `GEMPIC_ERROR`, warnings via `amrex::Warning`;
  use AMReX assertion macros directly only where device code needs them. TODOs as `/// @todo`.
  New files start with the BSD-3 banner of existing files.

## GPU code

- Loops over fields and particles run on device (`amrex::ParallelFor` or Kokkos kernels with
  `AMREX_GPU_HOST_DEVICE` functions).
- nvcc rejects device lambdas inside generic lambdas or private/protected member functions: put
  kernels in functors or free function templates.

## Commits, merge requests and issues

- GitLab: https://gitlab.mpcdf.mpg.de/gempic/gempic, merge requests target `master`.
- Merge request descriptions follow `.gitlab/merge_request_templates/default.md` (also when
  created with `glab`): fill in Summary and Related issues (`closes #N`), and tick only the
  checklist items that were actually done.
- Issues follow `.gitlab/issue_templates/default.md`: fill in Summary and the bug or the feature
  part, drop the other.
- Short imperative subjects (`Fix swapped tolerances in make_conjugate_gradient`).
- Label what you write: a `Co-Authored-By:` trailer naming your tool in commits, a last line
  naming it in MR and issue descriptions, comments and other messages.
