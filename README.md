# gempic

New gempic code based on AMReX

Installing with scripts
=====================
The library can be compiled with scripts available in the folder scripts. The available scripts are for supermuc, the mpcdf cluster and a desktop pc. Note that you might need to change the script for your desktop pc.

Installing manually
=====================

Install AMReX first enabeling particles (for each dimension):  
0.) cd build/amrex3D  
1.) cmake -D AMReX_SPACEDIM=3 -D AMReX_PARTICLES=ON ~/Documents/Codes/amrex/  
2.) make install  
3.) Run a cmake configuration that specifies the directory where AMReX is installed and the directory where the gempic code is located  

cmake -D AMReX_ROOT=~/Documents/Codes/amrex/installdir -D CMAKE_C_COMPILER=mpicc -D CMAKE_CXX_COMPILER=mpicxx ~/Documents/Codes/gempic


4.) Build the library  
make

5.) Generate Doxygen documentation (if desired)  
make doxygen

Changing dimension
=====================
cd ~build/amrex1D  
make install  
cd ~build/gempic  
make

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
