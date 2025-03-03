ran# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import yt
from yt.frontends.boxlib.data_structures import AMReXDataset

# %%
# read times series of fields
ts = yt.load('../../build/cpu-release-3D/Testing/FullDiagnostics/plt_field??????')
# Print number of items in time series
ntz = ts.__len__()
print('number of time slices ', ntz)
# print field list of first time steps. They are the same for all time steps
print("yt field list ",ts[0].field_list)

# %%
iter = 10
ds = ts[iter]
print(ds.field_list)
print(ts.params)

time = float(ds.current_time)
x_left = np.array(ds.domain_left_edge)
x_right = np.array(ds.domain_right_edge)
Ndim = ds.domain_dimensions
L = x_right - x_left
volume = L[0]*L[1]*L[2]
#yt.SlicePlot(ds, 'y', "eSolutionx")
yt.SlicePlot(ds, 'y', "Ex")

# %% with numpy
ds = ts[-1]
field_data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions )
E_x = field_data['boxlib','Ex']
E_sol = field_data['boxlib','eSolutionx']
E_err = field_data['boxlib','eErrorx']
plt.plot(E_err[0,0,:])
#plt.plot(E_sol[0,0,:])
#plt.plot(E_x[0,0,:])

# %%
for i in [30]:#range(0, len(ts), len(ts)//10):
    ds = ts[i]
    field_data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions )
    E_x = field_data['boxlib','Ex']
    E_sol = field_data['boxlib','eSolutionx']
    E_err = field_data['boxlib','eErrorx']
    plt.plot(E_x[0,0,:], c="tab:blue")
   # plt.plot(E_x[0,0,:], c="tab:blue")
    plt.plot(E_sol[0,0,:], '--', c="tab:orange")
   # plt.plot(E_sol[0,0,:], c="tab:orange")

#---------------------------------------------------------
# %%
time = float(ds.current_time)
x_left = np.array(ds.domain_left_edge)
x_right = np.array(ds.domain_right_edge)
Ndim = ds.domain_dimensions
L = x_right - x_left
volume = L[0]*L[1]*L[2]
#yt.SlicePlot(ds, 'y', "eSolutionx")
yt.SlicePlot(ds, 'y', "Jx")

# %% with numpy
ds = ts[-1]
field_data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions )
J_x = field_data['boxlib','Jx']
J_sol = field_data['boxlib','jSolutionx']
J_err = field_data['boxlib','jErrorx']
plt.plot(J_err[0,0,:])
#plt.plot(E_sol[0,0,:])
#plt.plot(E_x[0,0,:])

# %%
for i in range(0, len(ts), len(ts)//10):
    ds = ts[i]
    field_data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions )
    J_x = field_data['boxlib','Jx']
    J_sol = field_data['boxlib','jSolutionx']
    J_err = field_data['boxlib','jErrorx']
    plt.plot(J_x[0,0,:], c="tab:blue", lw=5-4.5*i/len(ts))
   # plt.plot(E_x[0,0,:], c="tab:blue")
    plt.plot(J_sol[0,0,:], '--', c="tab:orange", lw=5-4.5*i/len(ts))
   # plt.plot(E_sol[0,0,:], c="tab:orange")

#By---------------------------------------------------------
# %%
time = float(ds.current_time)
x_left = np.array(ds.domain_left_edge)
x_right = np.array(ds.domain_right_edge)
Ndim = ds.domain_dimensions
L = x_right - x_left
volume = L[0]*L[1]*L[2]
#yt.SlicePlot(ds, 'y', "eSolutionx")
yt.SlicePlot(ds, 'y', "By")

# %% with numpy
ds = ts[-1]
field_data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions )
B_y = field_data['boxlib','By']
B_sol = field_data['boxlib','bSolutiony']
B_err = field_data['boxlib','bErrory']
plt.plot(B_err[0,0,:])
#plt.plot(E_sol[0,0,:])
#plt.plot(E_x[0,0,:])

# %%
for i in  range(0, len(ts), len(ts)//10):
    ds = ts[i]
    field_data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions )
    B_y = field_data['boxlib','By']
    B_sol = field_data['boxlib','bSolutiony']
    B_err = field_data['boxlib','bErrory']
    plt.plot(B_y[0,0,:], c="tab:blue")
   # plt.plot(E_x[0,0,:], c="tab:blue")
    plt.plot(B_sol[0,0,:], '--', c="tab:orange")
   # plt.plot(E_sol[0,0,:], c="tab:orange")

# %%
