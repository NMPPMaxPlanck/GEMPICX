#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:32:28 2019

@author: irga
"""
import matplotlib.pylab as plt

import numpy as np

import yt

##########  1D   #####################

variable = 'Jx' # rho,  Ex, Ey, Ez, Jx, Jy, Jz
component = 'x' # x, y, z
slice_num = [0, 0] # 0, 1, ..., 7
timestep = 0 # 0, 1, ..., nsteps

####
ds = yt.load( '/home/irga/build/gempic/SliceData/' + variable + '_plotfile_' + str(timestep)) # Create a dataset object
ad = ds.all_data()  # extracts data to 1d array
data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions ) # extracts data to 3d array
values = data['boxlib', variable]
x=np.linspace(ds.domain_left_edge.v[0],ds.domain_right_edge.v[0],ds.domain_dimensions[0])
if component == 'x':
    plt.plot(x, values[:,slice_num[0],slice_num[1]])
elif component == 'y':
    plt.plot(x, values[slice_num[0],:,slice_num[1]])
elif component == 'z':
    plt.plot(x, values[slice_num[0],slice_num[1],:])
plt.show()


##########  2D   #####################

variable = 'rho' # rho,  Ex, Ey, Ez, Jx, Jy, Jz
components = 'xy' # xy, xz, yz
slice_num = 0 # 0, 1, ..., 7
timestep = 0 # 0, 1, ..., nsteps

####
ds = yt.load( '/home/irga/build/gempic/SliceData/' + variable + '_plotfile_' + str(timestep)) # Create a dataset object
ad = ds.all_data()  # extracts data to 1d array
data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions ) # extracts data to 3d array
values = data['boxlib', variable]

if components == 'xy':
    tst = 1
    plt.imshow(values[:,:,slice_num], cmap='hot', interpolation='nearest')
elif component == 'xz':
    plt.imshow(values[:,slice_num,:], cmap='hot', interpolation='nearest')
elif component == 'z':
    plt.imshow(values[slice_num,:,:], cmap='hot', interpolation='nearest')
plt.show()
