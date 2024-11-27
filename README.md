# GEMPIC

New gempic code based on [AMReX](https://github.com/AMReX-Codes/amrex)

[![Documentation](https://img.shields.io/badge/documentation-006c66)](https://gempic.pages.mpcdf.de/gempic/)
[![Master Pipeline Status](https://gitlab.mpcdf.mpg.de/gempic/gempic/badges/master/pipeline.svg)](https://gitlab.mpcdf.mpg.de/gempic/gempic/pipelines/master/latest)

## Requirements
- [AMReX](https://github.com/AMReX-Codes/amrex)
- [CMake](https://cmake.org/cmake/help/latest/index.html)
- [numdiff](https://www.nongnu.org/numdiff/) (To be deprecated. See issue #99)
- Documentation:
  - [Doxygen](www.doxygen.org) 
  - Python 3
    - [Sphinx](https://www.sphinx-doc.org)
    - [Breathe](https://breathe.readthedocs.io/en/latest/)
- Postprocessing:
  - Python 3
  - [Graphviz](http://www.graphviz.org/) (for graphical visualization of objects)


## Building with Presets
For systems on which the GEMPIC code is run regularly by multiple users we provide [cmake-presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html). So far this is implemented for the Raven Cluster of the MPCDF, Clang for MacOS and GCC for Linux. These presets are defined in the [CMakePresets](./CMakePresets.json) file. The presets might work on other systems as well, but this is not guaranteed.

All available presets can be listed using
```sh
cmake --list-presets
```

If more specific presets are required to build on systems which are not defined we recommend storing these presets in a `CMakeUserPresets.json` file.

Building GEMPIC using presets can be done by following the instructions below. Use different build directories for different presets.
```
cd /PATH/TO/GEMPIC
cmake --preset name-of-your-preset -S . -B /PATH/TO/BUILD/DIR
cmake --build /PATH/TO/BUILD/DIR
```
If multiple processes are available on the machine on which GEMPIC is build add the `-j <NProcs>` flag to the `cmake --build` command to parallelise the build. `<NProcs>` should be equal to the number of processes which should run in parallel.

## Personal Build Configuration

Only recommended for experienced users familiar with [CMake](https://cmake.org/cmake/help/latest/index.html). A more detailed instruction will be provided in an upcoming commit.
```sh
cd /PATH/TO/GEMPIC/PROJECT/DIR
mkdir build
cmake -S . -B build -D <AddCMakeOptions>
cmake --build ./build
```
The build process can be  accelerated by adding `-j <NProcs>` to `cmake --build` command. 
Do not use a number larger than the number of cores of the systems processor.

## CMake Options
The following options can be added to the personal build configuration of CMake
using `-D GEMPIC_OPTION Argument`

| CMake Option                 | Description                                                                                      | Default Value |
|------------------------------|--------------------------------------------------------------------------------------------------|---------------|
| `AMReX_SPACEDIM`             | The dimension of the simulation (`1`, `2` or `3`)                                                | `3`           |
| `GEMPIC_USE_CUDA`            | Use CUDA Backend of AMReX                                                                        | `OFF`         |
| `GEMPIC_USE_OMP`             | Use OpenMP Backend of AMReX <br>  (Not recommended due to reduced performance)                   | `OFF`         |
| `GEMPIC_USE_CUDA`            | Use CUDA Backend of AMReX                                                                        | `OFF`         |
| `GEMPIC_USE_LTO`             | Use Link-Time Optimization <br> (Turning it off reduces compile time, but may reduce performance)| `ON`          |
| `GEMPIC_BUILD_TESTS`         | Build the tests. Currently only toggles unit tests.                                              | `ON`          |
| `GEMPIC_BUILD_EXAMPLES`      | Build the examples. Currently only toggles all or none.                                          | `ON`          |
| `GEMPIC_BUILD_DOCUMENTATION` | Build the documentation. Only builds if depencies are met.                                       | `ON`          |


# Quickstart Example

The quickstart example applies the above guidelines for a specific simple configuration namely a 3D simulation for a CPU.

## Download and Build

Clone the gempic repository from gitlab and build the code using CMake. 
Before building make sure that all dependencies are satisfied.

```sh
git clone git@gitlab.mpcdf.mpg.de:gempic/gempic.git
cd gempic
cmake --preset cpu-release-3D -S .
cmake --build build/cpu-release-3D
```

## Run a simulation

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

## Plot simple diagnostics
Some python scripts (can be converted jupyter notebooks) can be found in `Examples` and adapted to the user run.
Just do (still in the `gempic_quickstart` run folder)

```sh
cp ../../Examples/Electrostatic/LandauVP.py .
python3 LandauVP.py
```

# Settings for vscode

- Install extensions: C/C++, C/C++ Extension Pack, Clang-Format, CMake, CMake Tools
- setup the environment: 
  - Open folder where the cloned gempic is. This will be $(workspaceFolder). Create a folder gempic_obj in the parent directory
  - Open settings from the menu 
    - In Extensions/CMake Tools: 
      - set build directory to ${workspaceFolder}/../gempic_obj 
      - set Cmake: Source Directory to ${workspaceFolder}
- formatting: set C_Cpp.clang_format_style to .clang-format (the .clang-format file is in the home of the gempic repository)
- Debugging: Two configuration examples are in scripts/launch.json
  - vlasov-maxwell: simulation example with input file
  - (lldb) Cmake: debugs the target that is set in vscode (without input file). Can be used to debug ctests

# Using QT-creator with gempic
Steps to set up gempic in QT-creator:  
1.) In QT-creator: File -> Open file or project -> click on CMakeLists.txt file  
2.) On Sidebar, click Projects and set correct build directory  
3.) If project doesn't compile automatically: close qccreator and reopen it, load project  
