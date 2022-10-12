#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This script can be used to 
# 1) evaluate the norm of the fields of a GEMPIC simulation
# 2) compare the results to the reference result "Weibel_01" given in the git repository (or other results)

###############################################################################
# Set parameters

# Field to exaluate (options: E1, E2, E3, B1, B2, B3, kin, mom1, mom2, mom3)
field = 'B3'
# Location of file with simulation output (need to adapt path)
filename = "/home/irene/Documents/jupyter_python/PIC_save_tmp_2.output"
# Location of file with reference solution (need to adapt path, available in git)
filename_ref = "/home/irene/Documents/Codes/gempic/simulations/PIC/reference_results/Weibel_01.txt"
# Interval to make subplot of
subpl = [3500,4500]

###############################################################################
# 1) Evaluate 1 simulation

# Importing and preparing necessary libraries
import matplotlib.pylab as plt
import numpy as np

# Read in values from simulation
n_lines = sum(1 for line in open(filename))

t = np.empty(n_lines-1)
E1 = np.empty(n_lines-1)
E2 = np.empty(n_lines-1)
E3 = np.empty(n_lines-1)
B1 = np.empty(n_lines-1)
B2 = np.empty(n_lines-1)
B3 = np.empty(n_lines-1)
kin = np.empty(n_lines-1)
mom1 = np.empty(n_lines-1)
mom2 = np.empty(n_lines-1)
mom3 = np.empty(n_lines-1)

file = open(filename, 'r')

line_num = 0
for line in file.readlines():
    if line_num > 0 :
        vals = line.rstrip().split(' ')
        t[line_num-1] = vals[0]
        E1[line_num-1] = vals[1]
        E2[line_num-1] = vals[2]
        E3[line_num-1] = vals[3]
        B1[line_num-1] = vals[4]
        B2[line_num-1] = vals[5]
        B3[line_num-1] = vals[6]
        kin[line_num-1] = vals[7]
        mom1[line_num-1] = vals[8]
        mom2[line_num-1] = vals[9]
        mom3[line_num-1] = vals[10]
    line_num = line_num + 1

exec("plt.semilogy(t," + field + ",'r-')")

###############################################################################
# 2) Compare simulation to reference result
    
# Read in values from reference solution
n_lines_ref = sum(1 for line in open(filename_ref))

t_ref = np.empty(n_lines_ref-1)
E1_ref = np.empty(n_lines_ref-1)
E2_ref = np.empty(n_lines_ref-1)
E3_ref = np.empty(n_lines_ref-1)
B1_ref = np.empty(n_lines_ref-1)
B2_ref = np.empty(n_lines_ref-1)
B3_ref = np.empty(n_lines_ref-1)
kin_ref = np.empty(n_lines_ref-1)
mom1_ref = np.empty(n_lines_ref-1)
mom2_ref = np.empty(n_lines_ref-1)
mom3_ref = np.empty(n_lines_ref-1)

file = open(filename_ref, 'r')

line_num = 0
for line in file.readlines():
    if line_num > 0 :
        vals = line.rstrip().split(' ')
        t_ref[line_num-1] = vals[0]
        E1_ref[line_num-1] = vals[1]
        E2_ref[line_num-1] = vals[2]
        E3_ref[line_num-1] = vals[3]
        B1_ref[line_num-1] = vals[4]
        B2_ref[line_num-1] = vals[5]
        B3_ref[line_num-1] = vals[6]
        kin_ref[line_num-1] = vals[7]
        mom1_ref[line_num-1] = vals[8]
        mom2_ref[line_num-1] = vals[9]
        mom3_ref[line_num-1] = vals[10]
    line_num = line_num + 1
    
dt_scale = int(np.floor(t[1]/t_ref[1]))

# Plot both Bz fields
min_n_lines = np.min([n_lines, n_lines_ref])

plt.figure()
exec("plt.semilogy(t[0:(min_n_lines-1)]," + field + "[0:(min_n_lines-1)],'r-', t_ref[0:dt_scale*(min_n_lines-1)]," + field + "_ref[0:dt_scale*(min_n_lines-1)],'b-')")

# Make subplot
plt.figure()
exec("plt.semilogy(t[subpl[0]:subpl[1]]," + field + "[subpl[0]:subpl[1]],'r-',t_ref[dt_scale*subpl[0]:dt_scale*subpl[1]]," + field + "_ref[dt_scale*subpl[0]:dt_scale*subpl[1]],'b-')")

# Plot evolution of error up to beginning of subinterval
plt.figure()
exec("plt.semilogy(t[0:subpl[0]], np.abs(" + field + "[0:subpl[0]]-" + field + "_ref[0:subpl[0]]), '-')")
