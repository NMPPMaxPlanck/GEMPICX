#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 15:13:26 2022

@author: Katharina Kormann (with input from Yingzhe Li)

This file loads data for E_y and computes the spectrum in the x-t plain
Specify your path in line 22 and the parameters in lines 35-41
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import yt
from yt.frontends.boxlib.data_structures import AMReXDataset 

pathname = '.' # Add the path to your data folder
pathname_out = pathname + '/processed'
try:
    os.mkdir(pathname_out)
except(FileExistsError):
    pass
os.chdir(pathname)
sim_name = "rmode"
slices =os.listdir("Plotfiles")
slices.sort()
print(slices)

# specify the data size
nx = 128; # number of points in x direction
nt = 25300 # number of time frames
nts = 100; # frequency of time frames used
ntz = int(nt/nts) # number of used time frames
dt = 0.005 # specify the time step size used
domain_x = 64 # length of domain in x direction
deltax = 0.5 # grid size

Exfft = np.zeros([nx,ntz],dtype=complex);

# read in the data
for i in range(0, ntz) :
    step = f'{(i+1)*nts:05d}'
    plotfile = './Plotfiles/{}{}'.format(sim_name, step)
    ds = AMReXDataset(plotfile)
    data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions )
    Ex = np.array(data['boxlib','E_y'])
    exs=np.sum(np.sum(Ex,2),1); # we sum over the y and z components
    Exfft[:,i] = exs#np.fft.fft(exs);

step = int(slices[0][-5:])
time = float(ds.current_time)
x_left = np.array(ds.domain_left_edge)
x_right = np.array(ds.domain_right_edge)
L = x_right - x_left
print(time)
print(L[0],L[1],L[2])

# apply the hann filter
hann = np.hanning(nt/nts);

for i in range(0,nx) : 
    for j in range(0,ntz) :
        Exfft[i,j] = Exfft[i,j]*hann[j]

# FFT in time
Ext = np.zeros([nx,ntz],dtype=complex);

Ext = np.fft.fftn(Exfft)
#for i in range(0,nx) :
#    Ext[i,:] = np.fft.fft(Exfft[i,:])

a = np.transpose(np.abs(Ext))/np.abs(Ext).max()

[N,L] =a.shape # Bz is a 4 dimensional array stroing the values of B3 at all the time steps and the initial value. N equals the time steps plus 1,  (K, M, L) are cell numbers in (x, y, z) directions.

T = (N-1)*dt  # dt is the time step

#dat1=np.genfromtxt('yinzhe_dispersion.txt')
lvls = np.logspace(-4.0, 0, 20)
lvls = np.logspace(-6.0, 1, 20)
plt.cla()
Lmax = int(L/2)
Nmax = ntz
om = 2*np.pi/T*np.arange(Nmax)
k = 2*np.pi/domain_x*np.arange(Lmax)*deltax
plt.contourf(k, om, a[0:Nmax, 0:Lmax], cmap=cm.jet,norm=colors.LogNorm(), levels=lvls)
plt.colorbar()
plt.xlabel('wave number' r'$\ k \Delta x$',fontsize=13)
plt.ylabel('frequency' r'$ \ \omega$',fontsize=14)
