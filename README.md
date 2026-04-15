# GEMPICX

[![Documentation](https://img.shields.io/badge/documentation-006c66)](https://gempic.pages.mpcdf.de/gempic/)
[![gitlab](https://img.shields.io/badge/GitLab-FC6D26?logo=gitlab&labelColor=gray)](https://gitlab.mpcdf.mpg.de/gempic/gempic)
[![github-mirror](https://img.shields.io/badge/GitHub%20mirror-gray?logo=github&labelColor=gray)](https://github.com/NMPPMaxPlanck/GEMPICX)

[![License](https://img.shields.io/badge/license-BSD_3-green?logo=open-source-initiative)](LICENSE)

## Overview
GEMPICX stands for Geometric Electro-Magnetic Particle-In-Cell for eXascale. It is an open-source C++20 [AMReX](https://github.com/AMReX-Codes/amrex)-based framework for structure-preserving plasma physics numerics based on a geometric discretization related to Mimetic Finite Difference.  
Through a field discretization following the de Rham structure of Maxwell's equations and a particle-in-cell approach, equations of motions are derived from a discrete action principle.  
Degrees of freedom are defined as point-values, edge-, face- and volume integrals on a primal and its dual grid.  
For implementation details, see the [documentation](https://gempic.pages.mpcdf.de/gempic/)  
For mathematical details, see [references](#references).

A list of example simulations can be found in the [`Examples`](Examples) directory. The documentation for these can be found [here](https://gempic.pages.mpcdf.de/gempic/latex/Examples.html), but a brief overview is:

| Example Simulation                                                                                          | Directory                                       | Description                                                                               |
|-------------------------------------------------------------------------------------------------------------|-------------------------------------------------|-------------------------------------------------------------------------------------------|
| [Electrostatic](https://gempic.pages.mpcdf.de/gempic/latex/Examples.html#electrostatic)                     | [`Electrostatic`](Examples/Electrostatic)       | Vlasov-Poisson or quasi-neutral                                                           |
| [Vlasov-Maxwell](https://gempic.pages.mpcdf.de/gempic/latex/Examples.html#vlasov-maxwell-non-relativistic)  | [`VlasovMaxwell`](Examples/VlasovMaxwell)       | Non-relativistic or relativistic                                                          |
| Quasineutral Vlasov-Maxwell                                                                                 | [`QuasineuvralVM`](Examples/QuasineutralVM)     | Iterative HYPRE solvers                                                                   |
| Maxwell                                                                                                     | [`Maxwell`](Examples/Maxwell)                   | Implicit-Explicit Runge-Kutta, no particles                                               |
| Magnetohydrodynamics                                                                                        | [`MHD`](Examples/MHD)                           | Finite Volume Explicit Runge-Kutta, no particles                                          |
| Low Storage Runge-Kutta Vlasov-Maxwell                                                                      | [`LSRK`](Examples/LSRK)                         | Fully kinetic, drift-kinetic, or hybrid (drift-kinetic electrons with fully kinetic ions) |
| [Linear Vlasov-Maxwell](https://gempic.pages.mpcdf.de/gempic/latex/Examples.html#linearized-vlasov-maxwell) | [`LinVlasovMaxwell`](Examples/LinVlasovMaxwell) | The Vlasov-Maxwell equations linearized around a stationary solution                      |
| [Cold plasma](https://gempic.pages.mpcdf.de/gempic/latex/Examples.html#cold-plasma)                         | [`ColdPlasma`](Examples/ColdPlasma)             | Maxwell's equations with a lineraized fluid description for electrons                     |

GEMPICX is being developed in the department of Numerical Methods in Plasma Physics (NMPP)
led by Prof. Eric Sonnendruecker at the Max Planck Institute for Plasma Physics
in Garching, Germany.

# Building and Installing GEMPICX
### Requirements
- [AMReX](https://github.com/AMReX-Codes/amrex)
- [CMake](https://cmake.org/cmake/help/latest/index.html)
- Documentation:
  - [Doxygen](https://doxygen.org)
  - [Pandoc](https://pandoc.org/)
  - curl
  - Python 3
    - [Sphinx](https://sphinx-doc.org)
    - [Breathe](https://breathe.readthedocs.io/en/latest/)
- Postprocessing:
  - Python 3
  - [Graphviz](http://graphviz.org/) (for graphical visualization of objects)


## Building with Presets
For systems on which the GEMPICX code is run regularly by multiple users we provide [cmake-presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html). So far this is implemented for the Raven Cluster of the MPCDF, Clang for MacOS and GCC for Linux. These presets are defined in the [CMakePresets](./CMakePresets.json) file. The presets might work on other systems as well, but this is not guaranteed.

All available presets can be listed using
```sh
cmake --list-presets
```

If more specific presets are required to build on systems which are not defined we recommend storing these presets in a `CMakeUserPresets.json` file.

Building GEMPICX using presets can be done by following the instructions below. Use different build directories for different presets.
```
cd /PATH/TO/GEMPICX
cmake --preset name-of-your-preset -S . -B /PATH/TO/BUILD/DIR
cmake --build /PATH/TO/BUILD/DIR
```
If multiple processes are available on the machine on which GEMPICX is build add the `-j <NProcs>` flag to the `cmake --build` command to parallelise the build. `<NProcs>` should be equal to the number of processes which should run in parallel.

## Personal Build Configuration

Only recommended for experienced users familiar with [CMake](https://cmake.org/cmake/help/latest/index.html). A more detailed instruction will be provided in an upcoming commit.
```sh
cd /PATH/TO/GEMPICX/PROJECT/DIR
mkdir build
cmake -S . -B build -D <AddCMakeOptions>
cmake --build ./build
```
The build process can be  accelerated by adding `-j <NProcs>` to `cmake --build` command. 
Do not use a number larger than the number of cores of the systems processor.

## CMake Options
The following options can be added to the personal build configuration of CMake
using `-D GEMPIC_OPTION Argument`

| CMake Option                      | Description                                                                                      |  Default Value  |
|-----------------------------------|--------------------------------------------------------------------------------------------------|-----------------|
| `AMReX_SPACEDIM`                  | The dimension of the simulation (`1`, `2` or `3`)                                                | `3`             |
| `GEMPIC_USE_CUDA`                 | Use CUDA Backend of AMReX                                                                        | `OFF`           |
| `GEMPIC_USE_HIP`                  | Use HIP Backend of AMReX                                                                         | `OFF`           |
| `GEMPIC_USE_OMP`                  | Use OpenMP Backend of AMReX <br>  (Not recommended due to reduced performance)                   | `OFF`           |
| `GEMPIC_USE_HYPRE`                | Use Hypre solver for AMReX                                                                       | Conditional[^1] |
| `GEMPIC_USE_LTO`                  | Use Link-Time Optimization <br> (Turning it off reduces compile time, but may reduce performance)| `ON`            |
| `GEMPIC_BUILD_TESTS`              | Build the tests. Currently only toggles unit tests.                                              | `ON`            |
| `GEMPIC_BUILD_EXAMPLES`           | Build the examples. Currently only toggles all or none.                                          | `ON`            |
| `GEMPIC_BUILD_DOCUMENTATION`      | Build the documentation.                                                                         | `OFF`           |
| `GEMPIC_BUILD_ONLY_DOCUMENTATION` | Build only the documentation of GEMPICX                                                           | `OFF`           |
| `AMReX_TINY_PROFILE`              | Use the built in tiny profiler                                                                   | `OFF`           |
| `GEMPICX_PROFILE_KERNELS`         | Enable tiny profiling in host device functions when executed on host. Requires AMReX_TINY_PROFILE | `OFF`           |

[^1]: Conditional logic for the default value of `GEMPIC_USE_HYPRE`: <br> `OFF` if `AMReX_SPACEDIM=1` or `GEMPIC_USE_CUDA=ON` or `GEMPIC_USE_HIP=ON` <br> `ON` otherwise

## Quickstart Example

The quickstart example applies the above guidelines for a specific simple configuration namely a 3D simulation for a CPU.

### Download and Build

Clone the gempic repository from gitlab and build the code using CMake. 
Before building make sure that all dependencies are satisfied.

```sh
git clone git@gitlab.mpcdf.mpg.de:gempic/gempic.git
cd gempic
cmake --preset cpu-release-3D -S .
cmake --build build/cpu-release-3D
```

### Run a simulation

Example simulations and input files can be found in `Examples`. 
Create a directory from which the simulation can be started. We run the simulation in a subdirectory of `gempic` project directory which is not 
recommended for real production runs.
```sh
mkdir -p runs/gempic_quickstart
cd runs/gempic_quickstart
cp ../../build/cpu-release-3D/Examples/Electrostatic/electrostatic .
cp ../../Examples/Electrostatic/landauVP.input .
```
Now run the simulation using
```sh
./electrostatic landauVP.input 
```

### Plot simple diagnostics
Some python scripts (can be converted jupyter notebooks) can be found in `Examples` and adapted to the user run.
Just do (still in the `gempic_quickstart` run folder)

```sh
cp ../../Examples/Electrostatic/LandauVP.py .
python3 LandauVP.py
```

# References
|   |
|------- |
|Michael Kraus, Katharina Kormann, Philip J. Morrison, and Eric Sonnendrücker. GEMPIC: Geometric electromagnetic particle-in-cell methods. Journal of Plasma Physics, 83(4), (2017). http://dx.doi.org/10.1017/S002237781700040X|
|B. Perse, K. Kormann, E. Sonnendrücker, Geometric Particle-in-Cell Simulations of the Vlasov–Maxwell System in Curvilinear Coordinates. SIAM Journal on Scientific Computing, 43(1), B194-B218, (2021). http://dx.doi.org/10.1137/20M1311934, https://arxiv.org/pdf/2002.09386.pdf|
|K. Kormann, E. Sonnendrücker. Energy-conserving time propagation for a structure-preserving particle-in-cell Vlasov–Maxwell solver. Journal of Computational Physics, 425, 109890, (2021). http://dx.doi.org/10.1016/j.jcp.2020.109890, https://arxiv.org/pdf/1910.04000.pdf|
|M. Campos Pinto, K. Kormann, E. Sonnendrücker. Variational Framework for Structure-Preserving Electromagnetic Particle-In-Cell Methods. J Sci Comput 91, 46 (2022). http://dx.doi.org/10.1007/s10915-022-01781-3, https://arxiv.org/pdf/2101.09247.|
|K. Kormann, E. Sonnendrücker. A Dual Grid Geometric Electromagnetic Particle in Cell Method. J Sci Comput 46, 5 (2024) https://doi.org/10.1137/23M1618910|
|G. Meng, K. Kormann, E. Poulsen, E. Sonnendrücker (2025). A geometric Particle-In-Cell discretization of the drift-kinetic and fully kinetic Vlasov–Maxwell equations. _Plasma Physics and Controlled Fusion_, _67_(5), 055007. https://dx.doi.org/10.1088/1361-6587/adc832|
