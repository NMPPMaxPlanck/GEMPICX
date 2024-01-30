# GEMPIC

New gempic code based on [AMReX](https://github.com/AMReX-Codes/amrex)

## Requirements
- [AMReX](https://github.com/AMReX-Codes/amrex)
- [CMake](https://cmake.org/cmake/help/latest/index.html)
- Documentation:
  - Doxygen 
  - Sphinx
  - [Breathe](https://breathe.readthedocs.io/en/latest/)
- Postprocessing:
  - [Graphviz](http://www.graphviz.org/) (for graphical visualization of objects)


## Building with Presets
For systems on which the GEMPIC code is run regularly by multiple users we provide [cmake-presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html). So far this is implemented for the Raven Cluster of the MPCDF, Clang for MacOS and GCC for Linux. These presets are defined in the [CMakePresets](./CMakePresets.json) file. The presets might work on other systems as well, but this is not guaranteed.

All available presets can be listed using
```sh
cmake --list-presets
```

If more specific presets are required to build on systems which are not defined we recommened to store these presets in a `CMakeUserPresets.json` file.

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

| CMake Option       | Description                                  | Default Value |
|--------------------|----------------------------------------------|---------------|
| `AMReX_SPACEDIM`   | The dimension of the simulation (`1`, `2` or `3`)  | `3`           |
| `GEMPIC_USE_CUDA`  | Use CUDA Backend of AMReX                    | `OFF`         |
| `GEMPIC_USE_OMP`   | Use OpenMP Backend of AMReX <br>  (Not recommended due to reduced performance)               | `OFF`         |


# Quickstart Example

## Download and Build

Clone the gempic repository from gitlab and build the code using CMake. 
Before building make sure that all dependencies are satisfied.

```sh
git clone git@gitlab.mpcdf.mpg.de:gempic/gempic.git
cd gempic
cmake -S . -B build
cmake --build build
```

## Run a simulation

Example simulations and input files can be found in `gempic/src/simulations`. 
From the `gempic` directory, do
```sh
mkdir runs/gempic_quickstart
cd runs/gempic_quickstart
cp ../../src/simulations/testing/IOFiles_3D/test_bernstein_waves_input.input .
```
Optionally, change the value of the parameter `time_loop.n_steps` in the input file `test_bernstein_waves_input.input', and run with
```sh
../../build/src/simulations/testing/test_bernstein_waves test_bernstein_waves_input.input 
```

## Plot simple diagnostics

Some jupyter notebooks can be found in 
`gempic/post_processing` and adapted to the user run.
Just do (still in the `gempic_quickstart` run folder)

```sh
cp ../../post_processing/rho_slices.ipynb .
jupyter lab rho_slices.ipynb         
```

# CTest

- To write a test code, write a main program in the testing directory corresponding to the code you are testing. This code should be in a file *test_mytest.cpp* and write the results of the test in a file called *test_mytest.output*   
- The new test is automatically added to the list of ctests when a *test_mytest.expected_output* file is added in the directory IOFiles_1D, IOFiles_2D or IOFiles_3D in the testing directory containing *test_mytest.cpp*. When ctest is called the files *test_mytest.output* and *test_mytest.expected_output* are compared and the test is passed when they are identical up to round of errors that are set by parameters used in numdiff (these parameters are hard coded in define_ctest.cmake). Which directory of IOFiles_1D, IOFiles_2D or IOFiles_3D is used when ctest is looking for tests is determined at compile time by the value of AMReX_SPACEDIM.
- Assuming *build* is the directory in which Gempic is built, the executables of the tests are found in build/src/my_dir/testing and can be run directly without performing all the ctests. You can also run all the tests in a testing directory by executing ctest in build/src/my_dir/testing

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
Do NOT open amrex as a project with QT-creator. This will build a second instance of amrex and gempic will not be able to find either of them. Building gempic as a project automatically links it to amrex, so amrex doesn't need to be set up as project itself.

Steps to set up gempic in QT-creator:  
1.) In QT-creator: File -> Open file or project -> click on CMakeLists.txt file  
2.) On Sidebar, click Projects and set correct build directory  
3.) If project doesn't compile automatically: close qccreator and reopen it, load project  

# Output of simulation
The executable simulations/valsov\_maxwell/vlasov\_maxwell generates output. This information is written into a file called simulation\_name\_tmp.output every 5 steps in case the simulation is interrupted. Each row contains the following information for one step:  
time | Ex | Ey | Ez | Bx | By | Bz | kinetic energy | momentumx | momentumy | momentumz| gauss error  
where the 2-norm has been applied to all the fields. If the simulation is run for dimensions other than 3d3v, then the corresponding components are not part of the output.


# Formatting
The format of the code (identation, spacing, etc.) follows the WarpX conventions, which are automatically done by `clang-format`. To format your code before committing run:

`source scripts/sanitize_code.sh`

The formatting should be done with clang-format version 14.

# Documentation
[Documentation for Gempic](https://gempic.pages.mpcdf.de/gempic/)
The documentation is constructed using Sphinx, and its source code can be found in the `gempic/doc/sphinx` folder.
- To build the GEMPIC Sphinx documentation locally on your computer:
  - Install the required packages: `sphinx`, `breathe`, `sphinx_rtd_theme`, `pandoc`
  - Execute `make html` in the `gempic/doc/sphinx` folder
  - The generated HTML files can be found in the `_build/html/` folder. Open `index.html` in your browser to view the documentation.

# Coding style and conventions
- we follow the [WarpX style convention](https://warpx.readthedocs.io/en/latest/developers/contributing.html#style-and-conventions)
