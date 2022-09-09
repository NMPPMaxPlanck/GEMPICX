#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 11:47:33 2019

@author: irga
"""

import matplotlib.pylab as plt

import numpy as np

import yt

#slc = yt.SlicePlot(ds, "z", "rho", center = [0,0,0])
components = ["x", "y", "z"]

########### rho ####################

ds = yt.load("/home/irga/build/gempic/SliceData/rho_plotfile")

#z
slc = yt.SlicePlot(ds, "z", "rho")
slc.show()
slc.save("/home/irga/build/gempic/slices/")

#y
slc = yt.SlicePlot(ds, "y", "rho")
slc.show()
slc.save("/home/irga/build/gempic/slices/")

#x
slc = yt.SlicePlot(ds, "x", "rho")
slc.show()
slc.save("/home/irga/build/gempic/slices/")

########### E ####################

for comp in components:
    for i in range(0, 9):
        ds = yt.load("/home/irga/build/gempic/SliceData/E" + comp + "_plotfile_" + str(i))
    
        #z
        slc = yt.SlicePlot(ds, "z", "E" + comp)
        slc.show()
        slc.save("/home/irga/build/gempic/slices/E" + comp + "/")
    
        #y
        slc = yt.SlicePlot(ds, "y", "E" + comp)
        slc.show()
        slc.save("/home/irga/build/gempic/slices/E" + comp + "/")
    
        #x
        slc = yt.SlicePlot(ds, "x", "E" + comp)
        slc.show()
        slc.save("/home/irga/build/gempic/slices/E" + comp + "/")
    
########### J ####################

for comp in components:
    for i in range(0, 9):
        ds = yt.load("/home/irga/build/gempic/SliceData/J" + comp + "_plotfile_" + str(i))
    
        #z
        slc = yt.SlicePlot(ds, "z", "J" + comp)
        slc.show()
        slc.save("/home/irga/build/gempic/slices/J" + comp + "/")
    
        #y
        slc = yt.SlicePlot(ds, "y", "J" + comp)
        slc.show()
        slc.save("/home/irga/build/gempic/slices/J" + comp + "/")
    
        #x
        slc = yt.SlicePlot(ds, "x", "J" + comp)
        slc.show()
        slc.save("/home/irga/build/gempic/slices/J" + comp + "/")

