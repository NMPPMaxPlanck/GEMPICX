#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import matplotlib.pylab as plt

import numpy as np

from tabulate import tabulate

# path needs to be adapted to build directory!
filename = "/home/irga/build/gempic/PIC_save.output"
filename_6d = "/home/irga/build/gempic/PIC_6dim.output"
filename_3d = "/home/irga/build/gempic/PIC_particle.output"
filename_weights = "/home/irga/build/gempic/weights.output"

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
plt.semilogy(t,E1,'-o')

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

########### pos at beginning ####################
    
n_lines_part = sum(1 for line in open(filename_3d))

x = np.empty(n_lines_part)
y = np.empty(n_lines_part)
z = np.empty(n_lines_part)

file = open(filename_3d, 'r')

line_num = 0
for line in file.readlines():
    vals = line.rstrip().split(',') #using rstrip to remove the \n
    x[line_num] = vals[0]
    y[line_num] = vals[1]
    z[line_num] = vals[2]
    line_num = line_num + 1
    
plt.scatter(x, y)
plt.scatter(x, z)
plt.scatter(y, z)

########### weights ####################
    
n_lines_w = sum(1 for line in open(filename_weights))

weights = np.empty(n_lines_w)
x1 = np.empty(n_lines_w)

file = open(filename_weights, 'r')

line_num = 0
for line in file.readlines():
    vals = line.rstrip().split(',') #using rstrip to remove the \n
    weights[line_num] = vals[0]
    x1[line_num] = vals[1]
    line_num = line_num + 1
    
n_lines_w
np.sum(weights)
plt.hist(x1, bins = 100)

########### compute weights with python and compare ####################


def weight_fun(x):
  return((1.0 + 0.5*np.cos(0.5*x))/n_lines_w*0.5/(2*np.pi))
  
n_lines_w = sum(1 for line in open(filename_weights))
weights = np.empty(n_lines_w)
x1 = np.empty(n_lines_w)
weight_diff = np.empty(n_lines_w)

file = open(filename_weights, 'r')

line_num = 0
for line in file.readlines():
    vals = line.rstrip().split(',') #using rstrip to remove the \n
    weights[line_num] = vals[0]
    x1[line_num] = vals[1]
    line_num = line_num + 1
    
weight_diff = np.abs(weights-weight_fun(x1))
    
plt.scatter(weights, weight_fun(x1))
plt.xlim(0, 2.4e-07)
plt.ylim(0, 2.4e-07)

