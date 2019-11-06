#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import matplotlib.pylab as plt

import numpy as np

from tabulate import tabulate

# path needs to be adapted to build directory!
filename = "/home/irga/build/gempic/PIC_save.output"
filename_6d = "/home/irga/build/gempic/PIC_6dim.output"

########### energy and mom ####################

n_lines = sum(1 for line in open(filename))

t = np.empty(n_lines)
E1 = np.empty(n_lines)
E2 = np.empty(n_lines)
E3 = np.empty(n_lines)
kin = np.empty(n_lines)
mom1 = np.empty(n_lines)
mom2 = np.empty(n_lines)
mom3 = np.empty(n_lines)
table = np.empty([n_lines, 8])

file = open(filename, 'r')

line_num = 0
for line in file.readlines():
    vals = line.rstrip().split(',') #using rstrip to remove the \n
    table[line_num,] = vals
    t[line_num] = vals[0]
    E1[line_num] = vals[1]
    E2[line_num] = vals[2]
    E3[line_num] = vals[3]
    kin[line_num] = vals[4]
    mom1[line_num] = vals[5]
    mom2[line_num] = vals[6]
    mom3[line_num] = vals[7]
    line_num = line_num + 1

print(tabulate(table,headers=['t', 'EEx', 'EEy', 'EEz', 'Ekin', 'mx', 'my', 'mz']))
plt.plot(t,E1+E2+E3+kin,'-o',t,mom1+mom2+mom3,'-*')

########### pos and vel ####################

n_lines_part = sum(1 for line in open(filename_6d))
n_part = n_lines_part/n_lines # remove

x = np.empty(n_lines_part)
y = np.empty(n_lines_part)
z = np.empty(n_lines_part)
vx = np.empty(n_lines_part)
vy = np.empty(n_lines_part)
vz = np.empty(n_lines_part)

file = open(filename_6d, 'r')

line_num = 0
for line in file.readlines():
    vals = line.rstrip().split(',') #using rstrip to remove the \n
    x[line_num] = vals[0]
    y[line_num] = vals[1]
    z[line_num] = vals[2]
    vx[line_num] = vals[3]
    vy[line_num] = vals[4]
    vz[line_num] = vals[5]
    line_num = line_num + 1
    
v_tot = abs(vx) + abs(vy) + abs(vz)

# histogram over timesteps
for i in range(0,n_lines-1):
    plt.figure()
    plt.hist(vx[range(i*n_part,(i+1)*n_part-1)], bins=50)
