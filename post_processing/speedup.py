#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 16:17:41 2020

@author: irga
"""

import numpy as np
import matplotlib.pylab as plt

small_procs = np.array([1,2,4,24])
small_time = np.array([23.9, 11.8, 6.6, 1.47])

big_procs = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32])
big_time = np.array([247.1, 126.8, 92.9, 69.0, 57.4, 49.3, 43.9, 39.0, 35.8, 32.3, 30.0, 28.2, 24.7, 23.4, 23.0, 20.3,19.9, 18.9, 18.6, 16.7, 16.4, 15.6, 15.1, 14.4, 13.9, 13.1, 17.9, 12.7, 16.2, 14.2, 15.7, 15.3])
big_speedup = big_time[0]/big_time
big_speedup_per_core = big_speedup/big_procs

plt.figure()
plt.plot(big_procs, big_time)
plt.figure()
plt.plot(big_procs,big_speedup,"-r", [1,32], [1,32],"-k")
plt.figure()
plt.plot(big_procs,big_speedup_per_core)

#plt.plot(small_procs, small_time, "-r", big_procs, big_time, "-b", [1,32], [1,32], "-k")
