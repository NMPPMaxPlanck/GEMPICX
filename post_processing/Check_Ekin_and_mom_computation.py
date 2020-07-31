#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This script compares the kinetic energy and momentum computed by gempic diagnostocs
# with results computed directly from the particle velocity information

 # Importing and preparing necessary libraries
import yt
import matplotlib.pylab as plt
import numpy as np
yt.funcs.mylog.setLevel(0)

###############################################################################
# Set parameters
# Domain length:
Length = 2*np.pi/1.25
# Path to Slices (necessary for python computation)
path =  '/home/irene/build/gempic/ParticlePlotfiles/Weibel'
# Path to gempic diagnostics
path_diag =  '/home/irene/build/gempic/PIC_save_tmp.output'
# Maximum timestep to load (reading all steps takes long, reduce number)
lim = 1100
step_width = 100
###############################################################################

# get dx
ds = yt.load(path + '00000')
dx = Length/ds.domain_dimensions[0]
dy = Length/ds.domain_dimensions[1]
dz = Length/ds.domain_dimensions[2]

# Turn available steps into strings
steps_num = range(0,lim,step_width)
steps = []
n_st = np.size(steps_num)
for i in range(n_st):
    steps.append('0'*(5-len(str(steps_num[i])))+str(steps_num[i]))
    
E_kin = np.empty(np.size(steps))
mom_x = np.empty(np.size(steps))
mom_y = np.empty(np.size(steps))
mom_z = np.empty(np.size(steps))

stp_ctr = 0

# Computing Ekin and mom for each step
for step in steps:
    # Loading YT data sets
    ds = yt.load(path + step)
    
    # Get Particle velocities and weights
    data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions )

    particle_v_x = data['electrons','particle_vx']
    particle_v_y = data['electrons','particle_vy']
    particle_v_z = data['electrons','particle_vz']
    particle_wei = data['electrons','particle_weight']
    
    # Compute kinetic energy
    E_kin[stp_ctr] = np.sum(0.5*particle_wei*(np.abs(particle_v_x)**2+np.abs(particle_v_y)**2+np.abs(particle_v_z)**2))
    
    # Compute momentum
    mom_x[stp_ctr] = np.sum(particle_wei*np.abs(particle_v_x))
    mom_y[stp_ctr] = np.sum(particle_wei*np.abs(particle_v_y))
    mom_z[stp_ctr] = np.sum(particle_wei*np.abs(particle_v_z))

    stp_ctr = stp_ctr+1
    
# Loading Ekin and mom from diagnostics file
t = np.empty(np.size(steps))
E_kin_diag = np.empty(np.size(steps))
mom_x_diag = np.empty(np.size(steps))
mom_y_diag = np.empty(np.size(steps))
mom_z_diag = np.empty(np.size(steps))

file = open(path_diag, 'r')

line_num = 0
entry = 0
for line in file.readlines():
    if (line_num-1) in steps_num :
        vals = line.rstrip().split(' ')
        t[entry] = vals[1]
        E_kin_diag[entry] = vals[7]
        mom_x_diag[entry] = vals[8]
        mom_y_diag[entry] = vals[9]
        mom_z_diag[entry] = vals[10]
        entry = entry+1
    line_num = line_num + 1

    
# Plot kinetic energies and difference
plt.plot(steps_num,E_kin,'-r',steps_num,E_kin_diag,'-b')
plt.figure()
plt.plot(steps_num,abs(E_kin-E_kin_diag),'-r')

# Plot momentum and differences
plt.figure()
plt.plot(steps_num,mom_x,'-r',steps_num,mom_x_diag,'-b')
plt.figure()
plt.plot(steps_num,abs(mom_x-mom_x_diag),'-r')
plt.figure()
plt.plot(steps_num,mom_y,'-r',steps_num,mom_y_diag,'-b')
plt.figure()
plt.plot(steps_num,abs(mom_y-mom_y_diag),'-r')
plt.figure()
plt.plot(steps_num,mom_z,'-r',steps_num,mom_z_diag,'-b')
plt.figure()
plt.plot(steps_num,abs(mom_z-mom_z_diag),'-r')
