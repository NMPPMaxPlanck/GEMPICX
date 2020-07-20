#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import matplotlib.pylab as plt

import numpy as np

from tabulate import tabulate

# path needs to be adapted to build directory!
filename = "/home/irga/build/gempic/PIC_save_tmp.output"
ref_filename = "/home/irga/Documents/Projects/gempic/simulations/PIC/reference_results/Weibel_01.txt"

filename_6d = "/home/irga/build/gempic/PIC_6dim.output"
filename_3d = "/home/irga/build/gempic/PIC_particle.output"
filename_weights = "/home/irga/build/gempic/weights.output"
filename_phi_init = "/home/irga/build/gempic/phi2.output"
filename_phi_end = "/home/irga/build/gempic/phi3.output"
filename_rho = "/home/irga/build/gempic/rho2.output"

########### ref energy and mom ####################
n_lines = sum(1 for line in open(ref_filename))
#sum(1 for line in open("/home/irga/Documents/Projects/gempic/simulations/PIC/reference_results/Weibel_01.txt"))

t_ref = np.empty(n_lines-1)
E1_ref = np.empty(n_lines-1)
E2_ref = np.empty(n_lines-1)
E3_ref = np.empty(n_lines-1)
B1_ref = np.empty(n_lines-1)
B2_ref = np.empty(n_lines-1)
B3_ref = np.empty(n_lines-1)
kin_ref = np.empty(n_lines-1)
mom1_ref = np.empty(n_lines-1)
mom2_ref = np.empty(n_lines-1)
mom3_ref = np.empty(n_lines-1)
table_ref = np.empty([n_lines, 11])

file = open(ref_filename, 'r')

line_num = 0
for line in file.readlines():
    if line_num > 0 :
        #print(line)
        #sys.stdout.flush()
        vals = line.rstrip().split(' ') #using rstrip to remove the \n
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


########### energy and mom ####################

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
table = np.empty([n_lines, 11])

file = open(filename, 'r')

line_num = 0
for line in file.readlines():
    if line_num > 0 :
        #print(line)
        #sys.stdout.flush()
        vals = line.rstrip().split(' ') #using rstrip to remove the \n
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

#print(tabulate(table,headers=['t', 'EEx', 'EEy', 'EEz', 'EBx', 'EBy', 'EBz', 'Ekin', 'mx', 'my', 'mz']))
#plt.plot(t,E1+E2+E3+B1+B2+B3+kin,'-o',t,mom1+mom2+mom3,'-*')
#plt.semilogy(t,B3,'r-', t_ref[0:(n_lines-1)],B3_ref[0:(n_lines-1)],'b-')
#plt.semilogy(t,np.pi*2/0.5*E1*0.5)
plt.semilogy(t,B3,'r-', t_ref[0:(n_lines-1)],B3_ref[0:(n_lines-1)],'b-')
plt.figure()
plt.semilogy(t,B3-B3_ref[0:(n_lines-1)])
    
plt.semilogy(t,0.5*B3,'r-', t,0.5*E1,'b-', t,0.5*E2,'g-', linewidth=0.5)
plt.ylim(1e-15, 1e3)
plt.xlim(0, 500) 

plt.figure()
plt.semilogy(t,B3,t,7e-8*np.exp(0.02784*2*t), linewidth=0.5)
plt.ylim((1e-13, 1e-2))
########### energy and mom (1D version) ####################

n_lines = sum(1 for line in open(filename))

t = np.empty(n_lines)
E = np.empty(n_lines)
B = np.empty(n_lines)
kin = np.empty(n_lines)
mom = np.empty(n_lines)
table = np.empty([n_lines, 5])

file = open(filename, 'r')

line_num = 0
for line in file.readlines():
    vals = line.rstrip().split(',') #using rstrip to remove the \n
    vals = vals[:-1]
    table[line_num,] = vals
    t[line_num] = vals[0]
    E[line_num] = vals[1]
    B[line_num] = vals[2]
    kin[line_num] = vals[3]
    mom[line_num] = vals[4]
    line_num = line_num + 1

#print(tabulate(table,headers=['t', 'EEx', 'EEy', 'EEz', 'EBx', 'EBy', 'EBz', 'Ekin', 'mx', 'my', 'mz']))
#plt.plot(t,E1+E2+E3+B1+B2+B3+kin,'-o',t,mom1+mom2+mom3,'-*')
plt.plot(t,E,'-')

########### rho and phi 1D ####################

n_lines = sum(1 for line in open(filename_phi_init))
rho = np.empty(n_lines)
phi_init = np.empty(n_lines)
phi_end = np.empty(n_lines)

file = open(filename_phi_init, 'r')

line_num = 0
for line in file.readlines():
    phi_init[line_num] = line.rstrip() #using rstrip to remove the \n
    line_num = line_num + 1

file = open(filename_phi_end, 'r')

line_num = 0
for line in file.readlines():
    phi_end[line_num] = line.rstrip() #using rstrip to remove the \n
    line_num = line_num + 1

file = open(filename_rho, 'r')

line_num = 0
for line in file.readlines():
    rho[line_num] = line.rstrip() #using rstrip to remove the \n
    line_num = line_num + 1

plt.plot(range(32),rho,'-')
plt.plot(range(32),phi_init,'-')
plt.plot(range(32),phi_end,'-')
plt.plot(range(32),abs(4*rho-phi_init),'-')

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
filename_3d = "/home/irga/build/gempic/test_particle_groups.output"
filename_3d = "/home/irga/build/gempic/test_particle_groups.output"
    
n_lines_part = sum(1 for line in open(filename_3d))
first_row = 1

x = np.empty(n_lines_part-first_row)
y = np.empty(n_lines_part-first_row)
z = np.empty(n_lines_part-first_row)

file = open(filename_3d, 'r')

line_num = 0
for line in file.readlines():
    if line_num > 0:
        vals = line.rstrip().split(',') #using rstrip to remove the \n
        x[line_num-first_row] = vals[0]
        y[line_num-first_row] = vals[1]
        z[line_num-first_row] = vals[2]
    line_num = line_num + 1
    
#plt.hist(x, bins = 8)
plt.scatter(x, y)
plt.scatter(x, z)
plt.scatter(y, z)

ind = 0.
plt.scatter(x[abs(z - 4.*np.pi/16.*(1./2.+ind))<0.1], y[abs(z - 4.*np.pi/16.*(1./2.+ind))<0.1])

########### vel at beginning ####################
filename_3v = "/home/irga/build/gempic/PIC_particleV.output"    
n_lines_part = sum(1 for line in open(filename_3d))

vx = np.empty(n_lines_part)
vy = np.empty(n_lines_part)
vz = np.empty(n_lines_part)

file = open(filename_3v, 'r')

line_num = 0
for line in file.readlines():
    vals = line.rstrip().split(',') #using rstrip to remove the \n
    vx[line_num] = vals[0]
    vy[line_num] = vals[1]
    vz[line_num] = vals[2]
    line_num = line_num + 1
    
plt.hist(vx, bins = 20)
plt.hist(vy, bins = 20)
plt.hist(vz, bins = 20)

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

