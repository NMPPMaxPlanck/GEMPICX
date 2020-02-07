#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:41:42 2020

@author: irga
"""

import matplotlib.pylab as plt

import numpy as np

import yt

from tabulate import tabulate

########### E_sol_x ################
ds = yt.load("home/irga/build/gempic/SliceDataMW/Ex_plotfile_1")
data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions ) # extracts data to 3d array
E = data['boxlib','E']
plt.plot(E[:,0])

ds = yt.load("/home/irga/build/gempic/SliceData/" + "B" + "x" + "_plotfile_" + str(1))
data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions )
Bx = data['boxlib','Bx']
plt.plot(Bx[:,0])

ds = yt.load("/home/irga/build/gempic/SliceData/" + "B_sol" + "x" + "_plotfile_" + str(1))
data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions )
B_solx = data['boxlib','B_solx']
plt.plot(B_solx[:,0])

########### rho ####################

ds = yt.load("/home/irga/build/gempic/SliceData2D/rho_plotfile_0_test")
data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions ) # extracts data to 3d array
rho = data['boxlib','rho']
#plt.plot(rho[:,0])

slc = yt.SlicePlot(ds, "z", "rho")
slc.show()
slc.save("/home/irga/build/gempic/slices2D/")

########### phi ####################
#init

ds = yt.load("/home/irga/build/gempic/SliceData2D/phi_init_plotfile_0")
data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions ) # extracts data to 3d array
phi_init = data['boxlib','phi_init']
plt.plot(phi_init[:,0])

slc = yt.SlicePlot(ds, "z", "phi_init")
slc.show()
slc.save("/home/irga/build/gempic/slices2D/")

#end

ds = yt.load("/home/irga/build/gempic/SliceData2D/phi_end_plotfile_0")
data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions ) # extracts data to 3d array
phi_end = data['boxlib','phi_end']
#plt.plot(phi_init[:,0])
#plt.plot(phi_end[:,0])

slc = yt.SlicePlot(ds, "z", "phi_end")
slc.show()
slc.save("/home/irga/build/gempic/slices2D/")

########### fields ####################
steps = range(0,2,1)
components = ["x", "y", "z"]
fields = ["E", "E_sol", "J"]

for fie in fields:
    for comp in components:
        for i in steps:
            ds = yt.load("/home/irga/build/gempic/SliceData/" + fie + comp + "_plotfile_" + str(i))
            slc = yt.SlicePlot(ds, "z", fie + comp)
            slc.show()
            slc.save("/home/irga/build/gempic/slices2D/" + fie + comp + "/")
            
components = ["x", "y", "z"]
fields = ["B", "B_sol"]

for fie in fields:
    for comp in components:
        for i in steps:
            ds = yt.load("/home/irga/build/gempic/SliceData/" + fie + comp + "_plotfile_" + str(i))
            slc = yt.SlicePlot(ds, "z", fie + comp)
            slc.show()
            slc.save("/home/irga/build/gempic/slices2D/" + fie + comp + "/")
            
# Ex[0]
ds = yt.load("/home/irga/build/gempic/SliceData/" + "E" + "x" + "_plotfile_" + "0")
data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions )
Ex = data['boxlib','Ex']
plt.plot(Ex[0,:])

ds = yt.load("/home/irga/build/gempic/SliceData/" + "E" + "y" + "_plotfile_" + "0")
########### energy and mom ####################
plt.semilogy(t,E1,'-')
max_step = 1998
filename = "/home/irga/build/gempic/tmp/PIC_save.output" + str(max_step)
#filename = "/home/irga/build/gempic/python_plots/Ampere/PIC_save.output" + str(max_step) + "_mainX"
n_lines = sum(1 for line in open(filename))

t = np.empty(n_lines)
E1 = np.empty(n_lines)
E2 = np.empty(n_lines)
B3 = np.empty(n_lines)
kin = np.empty(n_lines)
mom1 = np.empty(n_lines)
mom2 = np.empty(n_lines)
table = np.empty([n_lines, 7])

file = open(filename, 'r')

line_num = 0
for line in file.readlines():
    vals = line.rstrip().split(',') #using rstrip to remove the \n
    table[line_num,] = vals
    t[line_num] = vals[0]
    E1[line_num] = vals[1]
    E2[line_num] = vals[2]
    B3[line_num] = vals[3]
    kin[line_num] = vals[4]
    mom1[line_num] = vals[5]
    mom2[line_num] = vals[6]
    line_num = line_num + 1

#print(tabulate(table,headers=['t', 'EEx', 'EEy', 'EBz', 'Ekin', 'mx', 'my']))
#plt.plot(t,E1+E2+B3+kin,'-')
#plt.plot(t,E2,'-') 
plt.semilogy(t,E1,'-')

plt.semilogy(t,B3,'-')

########### energy and mom (1D) ####################
max_step = 1998
filename = "/home/irga/build/gempic/tmp/PIC_save.output" + str(max_step)
n_lines = sum(1 for line in open(filename))

t = np.empty(n_lines)
E1 = np.empty(n_lines)
B3 = np.empty(n_lines)
kin = np.empty(n_lines)
mom1 = np.empty(n_lines)
table = np.empty([n_lines, 5])

file = open(filename, 'r')

line_num = 0
for line in file.readlines():
    vals = line.rstrip().split(',') #using rstrip to remove the \n
    table[line_num,] = vals
    t[line_num] = vals[0]
    E1[line_num] = vals[1]
    B3[line_num] = vals[2]
    kin[line_num] = vals[3]
    mom1[line_num] = vals[4]
    line_num = line_num + 1

#print(tabulate(table,headers=['t', 'EEx', 'EEy', 'EBz', 'Ekin', 'mx', 'my']))
#plt.plot(t,E1+B3+kin,'-')
plt.semilogy(t,E1,'-') 

########### energy and mom (1D2V) ####################
max_step = 2097
filename = "/home/irga/build/gempic/tmp/PIC_save.output" + str(max_step)
n_lines = sum(1 for line in open(filename))

t = np.empty(n_lines)
E1 = np.empty(n_lines)
E2 = np.empty(n_lines)
B3 = np.empty(n_lines)
kin = np.empty(n_lines)
mom1 = np.empty(n_lines)
table = np.empty([n_lines, 7])

file = open(filename, 'r')

line_num = 0
for line in file.readlines():
    vals = line.rstrip().split(',') #using rstrip to remove the \n
    table[line_num,] = vals
    t[line_num] = vals[0]
    E1[line_num] = vals[1]
    E2[line_num] = vals[2]
    B3[line_num] = vals[3]
    kin[line_num] = vals[4]
    mom1[line_num] = vals[5]
    mom1[line_num] = vals[6]
    line_num = line_num + 1

#print(tabulate(table,headers=['t', 'EEx', 'EEy', 'EBz', 'Ekin', 'mx', 'my']))
#plt.plot(t,E1+B3+kin,'-')
plt.semilogy(t,E1,'-')

 ########### energy and mom 3D ####################
max_step = 22776 #22440
max_step = 65
#plt.semilogy(t[0:(max_step+1)],E1_1[0:(max_step+1)],'-')

filename = "/home/irga/build/gempic/tmp2/PIC_save.output" + str(max_step)
#filename = "/home/irga/Documents/Projects/gempic/simulations/PIC/reference_results/Weibel_00.txt"
#filename = "/home/irga/build/gTMP/tmp/PIC_save.output" + str(max_step)

#filename = "/home/irga/build/gempic/python_plots/Weibel/PIC_save.output22776"
#filename = "/home/irga/build/gempic/tmp/PIC_save.output22776"
#filename = "/home/irga/build/gempic/tmp/PIC_save.output_W22647"

n_lines = sum(1 for line in open(filename))

t = np.empty(n_lines)
E1 = np.empty(n_lines)
E2 = np.empty(n_lines)
B1 = np.empty(n_lines)
B2 = np.empty(n_lines)
E3 = np.empty(n_lines)
B3 = np.empty(n_lines)
kin = np.empty(n_lines)
mom1 = np.empty(n_lines)
mom2 = np.empty(n_lines)
mom3 = np.empty(n_lines)
table = np.empty([n_lines, 11])

file = open(filename, 'r')

line_num = 0
for line in file.readlines():
    vals = line.rstrip().split(',') #using rstrip to remove the \n
    table[line_num,] = vals
    t[line_num] = vals[0]
    E1[line_num] = vals[1]
    E2[line_num] = vals[2]
    E3[line_num] = vals[3]
    B1[line_num] = vals[4]
    B2[line_num] = vals[5]
    B3[line_num] = vals[6]
    kin[line_num] = vals[7]
    mom1[line_num] = vals[8]
    mom2[line_num] = vals[9]
    mom3[line_num] = vals[10]
    line_num = line_num + 1

plt.semilogy(t[0:(max_step+1)],E1_ref[0:(max_step+1)],'-')
plt.semilogy(t[0:(max_step+1)],E1[0:(max_step+1)],'-')

#print(tabulate(table,headers=['t', 'EEx', 'EEy', 'EBz', 'Ekin', 'mx', 'my']))
#plt.plot(t,E1+E2+B3+kin,'-')
#plt.plot(t,E2,'-') 
plt.semilogy(t,E1,'-')
#plt.semilogy(t,E2,'-')
#plt.semilogy(t,E3,'-')
#plt.semilogy(t,B1,'-')
#plt.semilogy(t,B2,'-')
plt.semilogy(t,B3,'-',t,3e-9*np.exp(0.018875511156200*2*t))
plt.semilogy(t,B3,'-',t,3e-7*np.exp(0.02784*2.2*t))
plt.ylim((1E-9,2E-3))
