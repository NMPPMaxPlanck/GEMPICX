#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This script can be used to evaluate the number of cells that the fastest particle traverses in one step
# in each direction

# Importing and preparing necessary libraries
import yt
import matplotlib.pylab as plt
import numpy as np
yt.funcs.mylog.setLevel(0)

###############################################################################
# Set parameters
# Maximum timestep to load (reading all steps takes long, reduce number)
lim = 1100
step_width = 100
# Path of Plotflies:
path = '/home/irene/build/gempic/ParticlePlotfiles/Weibel'
# Problem parameters
L = 4*np.pi
n_cell = np.array([24, 8, 8])
dt = 0.02
###############################################################################

# Turn available steps into strings
steps_num = range(0,lim,step_width)
steps = []
n_st = np.size(steps_num)
for i in range(n_st):
    steps.append('0'*(5-len(str(steps_num[i])))+str(steps_num[i]))
    
dx = L*1/n_cell

# Write Data (can take long, reduce lim if necessary)
x_max_n_cell = []
y_max_n_cell = []
z_max_n_cell = []
for i in range(n_st):
    ds = yt.load(path+steps[i])
    
    # Get Particle Positions and Velocities
    data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions )
    particle_v_x = data['electrons','particle_vx']
    particle_v_y = data['electrons','particle_vy']
    particle_v_z = data['electrons','particle_vz']
    
    #how many cells do we go in x direction?
    n_cell_x = particle_v_x*dt/dx[0]
    n_cell_y = particle_v_y*dt/dx[1]
    n_cell_z = particle_v_z*dt/dx[2]
    
    x_max_n_cell.append(np.max(n_cell_x))
    y_max_n_cell.append(np.max(n_cell_y))
    z_max_n_cell.append(np.max(n_cell_z))
#
    
plt.plot(range(n_st), x_max_n_cell, '-r', range(n_st), y_max_n_cell, '-b', range(n_st), z_max_n_cell, '-g')

