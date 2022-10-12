#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing and preparing necessary libraries
import yt
import matplotlib.pylab as plt
import numpy as np
yt.funcs.mylog.setLevel(0)

# This script plots and saved the particle positions for given timesteps

###############################################################################
# Set parameters
# Path to Slices
path =  '/home/irene/build/gempic/ParticlePlotfiles/Weibel'
# Path to save figures
path_fig =  '/home/irene/build/gempic/particles/plot'
# Maximum timestep to load (reading all steps takes long, reduce number)
lim = 1100
step_width = 100
###############################################################################

# Turn available steps into strings
steps_num = range(0,lim,step_width)
steps = []
n_st = np.size(steps_num)
for i in range(n_st):
    steps.append('0'*(5-len(str(steps_num[i])))+str(steps_num[i]))

for i in range(n_st):
    ds = yt.load(path+steps[i])
    
    # Get Particle Positions and Velocities
    data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions )
    particle_position_x = data['electrons','particle_position_x']
    particle_position_y = data['electrons','particle_position_y']
    particle_position_z = data['electrons','particle_position_z']
    
    plt.figure()
    plt.scatter(particle_position_x, particle_position_y, s = 0.1)
    plt.savefig(path_fig+steps[i], bbox_inches='tight')
    plt.close()
#