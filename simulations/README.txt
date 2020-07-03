To run one of the simulations:

1) Build gempic
2) Generate an input file following the structure in sample.input. Some input files are available in the PIC folder.
3) In the build directory: ./simulations/SIMULATION_FOLDER/SIMULATION INPUT_FILE_PATH (ex: ./simulation/PIC/PIC ~/Documents/Projects/gempic/simulations/PIC/PIC_params_standard.input )
3.1) Parallel: (ex: mpirun -np 4 ./simulation/PIC/PIC ~/Documents/Projects/gempic/simulations/PIC/PIC_params_standard.input )
4) The simulation will store results in file PIC_save_tmp.output during the simulation and in PIC_save.output at the end of The simulation

To run ctests
1) Build gempic
2) In build directory: ctest
3) For a specific test: ctest -R particles
