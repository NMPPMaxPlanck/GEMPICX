# gempic

New gempic code based on AMReX

Installing with scripts
=====================
The library can be compiled with scripts available in the folder scripts. The available scripts are for supermuc, the mpcdf cluster and a desktop pc. Note that you might need to change the script for your desktop pc.

Compiling and running on MPCDF machines
=======================================
- go into gempic/scripts directory
### On CPU
- type ./compile_mpcdf.sh
- a directory gempic_obj has been generated in your home directory. It contains all the tests and simulations.
- An example batch script run_mpcdf.sh has been created in gempic_obj. Copy it to your run directory and adapt as needed
### On Raven GPU
- type ./compile_gpu_raven-gcc.sh
- a directory gempic_gpu_obj has been generated in your home directory. It contains all the tests and simulations.
- Use run_gpu_mpcdf-gcc.sh from script directory. Copy it to your run directory and adapt as needed

Installing manually
=====================
- needed software: C++ compiler, MPI, cmake 

1.) git clone  
2.) git submodule init; git submodule update  
3.) cd gempic; mkdir build; cd build  
4.) cmake ..   
5.) make 

Main executable vlasov_maxwell is in build/simulation/vlasov_maxwell  

- Update AMReX version in submodule
1. cd third_party/amrex
2. git checkout SHA (commit that is desired)
3. cd ../..
4. git add third_party/amrex
5. git commit -m"update AMReX to ..."
6. git submodule update

To update your own version with the version from the repository type 
*git submodule update* 

Changing dimension
=====================
cd ~build/amrex1D  
make install  
cd ~build/gempic  
make

Ctests
======
- To write a test code, write a main program in the testing directory corresponding to the code you are testing. This code should be in a file *test_mytest.cpp* and write the results of the test in a file called *test_mytest.output*   
- The new test is automatically added to the list of ctests when a *test_mytest.expected_output* file is added in the directory IOFiles_1D, IOFiles_2D or IOFiles_3D in the testing directory containing *test_mytest.cpp*. When ctest is called the files *test_mytest.output* and *test_mytest.expected_output* are compared and the test is passed when they are identical up to round of errors that are set by parameters used in numdiff (these parameters are hard coded in define_ctest.cmake). Which directory of IOFiles_1D, IOFiles_2D or IOFiles_3D is used when ctest is looking for tests is determined at compile time by the value of AMReX_SPACEDIM.
- Assuming *build* is the directory in which Gempic is built, the executables of the tests are found in build/src/my_dir/testing and can be run directly without performing all the ctests. You can also run all the tests in a testing directory by executing ctest in build/src/my_dir/testing

Using QT-creator with gempic
=====================
Do NOT open amrex as a project with QT-creator. This will build a second instance of amrex and gempic will not be able to find either of them. Building gempic as a project automatically links it to amrex, so amrex doesn't need to be set up as project itself.

Steps to set up gempic in QT-creator:  
1.) In QT-creator: File -> Open file or project -> click on CMakeLists.txt file  
2.) On Sidebar, click Projects and set correct build directory  
3.) If project doesn't compile automatically: close qccreator and reopen it, load project  

Output of simulation
=====================
The executable simulations/valsov\_maxwell/vlasov\_maxwell generates output. This information is written into a file called simulation\_name\_tmp.output every 5 steps in case the simulation is interrupted. Each row contains the following information for one step:  
time | Ex | Ey | Ez | Bx | By | Bz | kinetic energy | momentumx | momentumy | momentumz| gauss error  
where the 2-norm has been applied to all the fields. If the simulation is run for dimensions other than 3d3v, then the corresponding components are not part of the output.

Project branches
=====================

|branch            | description |
|------------------|--------------------------------------------|
| `master`           | current main version of the code |
| `poisson_Epara`    | implemention of new Epara field solver |  
| `live_code`        | added code from the live coding sessions, not relevant to production, can be removed when all participants agree |
| `gpu_cuda_loop`    | replacing ParallelFor loops with cuda loops |
| `poisson_order`    | attempt at implementing an amrex-type solver that is edge/face centered |

Dependencies
=====================

- CMake
- Doxygen
- Graphviz ([for graphical visualization of objects](http://www.graphviz.org/))
- Sphinx
- Breathe ([for bridging between Sphinx and Doxygen documentation systems](https://breathe.readthedocs.io/en/latest/))

Formatting
=====================
The format of the code (identation, spacing, etc.) follows the WarpX conventions, which are automatically done by `clang-format`. To format your code before committing run:

`source scripts/sanitize_code.sh`

Documentation
=====================
[Documentation for Gempic](https://gempic.pages.mpcdf.de/gempic/)

