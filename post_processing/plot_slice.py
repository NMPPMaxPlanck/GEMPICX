#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 11:47:33 2019

@author: irga
"""

import matplotlib.pylab as plt

import numpy as np

import yt

ds = yt.load("/home/irga/build/gempic/rho_plotfile")

slc = yt.SlicePlot(ds, "z", "rho")
slc.show()
slc.save()
