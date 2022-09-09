import matplotlib.pylab as plt
import numpy as np
import yt

ds_prep = yt.load("/home/irene/build/gempic/simulations/vlasov_maxwell/rho_loop_prep")
ds_diagn = yt.load("/home/irene/build/gempic/simulations/vlasov_maxwell/rho_loop_diagn")

data_prep = ds_prep.covering_grid( 0, ds_prep.domain_left_edge, ds_prep.domain_dimensions )
data_diagn = ds_diagn.covering_grid( 0, ds_diagn.domain_left_edge, ds_diagn.domain_dimensions )


rho_prep = np.array(data_prep['boxlib','rho'])
rho_diagn = np.array(data_diagn['boxlib','rho'])

np.max(np.abs(rho_prep-rho_diagn))

slicey = 3
slicez = 3
plt.plot(rho_prep[:,slicey,slicez])
plt.plot(rho_diagn[:,slicey,slicez]+0.1)

# GEMPIC_maxwell_yee.H line 1492:
#amrex::Vector<std::string> varnames = {"rho"};
#WriteSingleLevelPlotfile("rho_loop_diagn", rho, varnames, ifr.geom, 0, 0);

#GEMPIC_loop_preparation.H line 90:
#amrex::Vector<std::string>  varnames = {"rho"};
#WriteSingleLevelPlotfile("rho_loop_prep", mw_yee->rho, varnames, infra.geom, 0, 0);

#vlasov_maxwell.cpp: set spline degree to 4

# General setting
sim_name = "Weibel"

# Grid parameters
n_cell_vector = 24 8 8
max_grid_size_vector = 4 4 4
is_periodic_vector = 1 1 1 # 1 -> periodic, 0 -> else

# Particle parameters
n_part_per_cell = 100 # number of particles per cell
charge = -1.0 # vector (one entry per species)
mass = 1.0 # vector (one entry per species)
num_gaussians = 1

# Gaussian parameters
velocity_mean_0 = 0.0 0.0 0.0
velocity_deviation_0 = 0.014142135623730949 0.04898979485566356 0.04898979485566356 # 0.02/sqrt(2) and sqrt(12)*0.02/sqrt(2)
velocity_weight_0 = 1.0 1.0 1.0

# wave vector
k = 1.25 1.25 1.25 # 1.25 for Weibel, 0.5 for Landau

# Simulation parameters
n_steps = 0
dt = 0.02
freq_x = 100001 
freq_v = 100001 
freq_slice = 100001 

# Function parameters
density = "1.0 + 0.0 * cos(kvarx * x)"
Bx = "0.0"
By = "0.0"
Bz = "1e-3 * cos(kvarx * x)"
phi = "4 * 0.5 * cos(0.5 * x)"

# Choose the propagator: 0 - Boris-FD, 1 - HS-FEM
propagator = 3

# Tolerances for iterative solver (if needed by the chosen propagator)
tolerance_particles = 1.e-10

# Restart parameters
restart = 0
checkpoint_file = "Checkpoint/Test_Weibel_2_"
curr_step = 4
