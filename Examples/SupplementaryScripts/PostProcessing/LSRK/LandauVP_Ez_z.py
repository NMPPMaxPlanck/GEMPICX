# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3.11.1 64-bit
#     language: python
#     name: python3
# ---

# %% [markdown]
# - Convert to jupyter notebook with 'jupytext --to ipynb LandauVP.py'
# - and back to python percent format with 'jupytext --to py:percent LandauVP.ipynb'

# %% [markdown]
#

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pandas as pd
import yt
#from yt.frontends.boxlib.data_structures import AMReXDataset 
#yt.enable_parallelism()
yt.set_log_level(0) # do not show log output

# %%
pathname = '.' # Notebook should be in directory where code is run
print('run directory: ', os.getcwd())
pathname_out = pathname + '/processed'
try:
    os.mkdir(pathname_out)
except(FileExistsError):
    pass
os.chdir(pathname)
# read times series
ts = yt.load('./FullDiagnostics/plt_field??????')
ntz = ts.__len__() # number of items in time series
# save times corresponding to each dataset
times = np.zeros((ntz),dtype=float) 
for i in range(ntz):
    times[i] = ts[i].current_time

# print field list and choose field to be used for dispersion relation
ds = ts[0]
print(ds.field_list,ds.domain_left_edge, ds.domain_right_edge, ds.domain_dimensions)


# %%
# read in the data for each time slice
field = 'Ez'
storage = {}
for store, ds in ts.piter(storage=storage):
    data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions )
    arr = np.array(data['boxlib',field])
    # store.result = np.sum(np.sum(arr,2),1); # we sum over the y and z components
    store.result = np.sum(np.sum(arr,0),0); # we sum over the x and y components
    time = float(ds.current_time)
    #print(time)
    
# %%
# get space dimensions
nx, ny, nz = ts[0].domain_dimensions
x_left = np.array(ds.domain_left_edge)
x_right = np.array(ds.domain_right_edge)
L = x_right - x_left
# fill array for FFT
arr = np.zeros([nz,ntz]);
print(arr.shape)
for data in storage.items():
    arr[:,data[0]] = data[1] / (nx*ny)   

arr1 = arr
# FFT in space
arrfft = np.zeros_like(arr,dtype=complex)
for i in range(ntz):
    arrfft[:,i] = np.fft.fft(arr[:,i])
# Shift and normalize array
np.fft.fftshift(arrfft)
arrfft = arrfft / nz

# plots 
# Parameters for exact solution
# least damped mode
epsilon = 0.04
r = 0.424666
phase = 0.3357725
coef = 2 * epsilon * r 
omegar = 1.2850
gamma = -0.0661

plt.figure()
fig, axs = plt.subplots(3, 1,sharex=True,tight_layout=True)
# Analytical solution of fundamental mode (real part and imaginary part)
if field == 'rho':
    axs[0].plot(times, 
            (coef*np.cos(omegar*times-phase)*np.exp(gamma*times))
            ,label="exact")
    axs[1].plot(times, np.zeros_like(times))
elif field == 'Ez':
    coef = 2 * epsilon * r    
    axs[0].plot(times, np.zeros_like(times))
    axs[1].plot(times, (coef*np.cos(omegar*times-phase)*np.exp(gamma*times)),label="exact")
# Analytical solution of fundamental mode (mudulus squared)    
axs[2].plot(times, np.log((coef*np.cos(omegar*times-phase)*np.exp(gamma*times))**2))

# Analytical solution with inverse Laplace transform
#anaLap=np.fromfile("/Users/sonnen/Codes/gempic_home/gempic/dispersion_relations/solLandauIAW.npy")
#print(times.shape,anaLap.shape)
#print(anaLap[16:])
#axs[0].plot(times, anaLap[16:]/6.28,'k--',label="anaLap")

# Numerical solution of fundamental mode and harmonics to check noise level
for i in range(1,4):
    axs[0].plot(times, arrfft[i,:].real,label="k="+str(i))
    axs[1].plot(times, arrfft[i,:].imag)
    axs[2].plot(times, 2*np.log(np.abs(arrfft[i,:])))
# Plot higher order contributions of exact solution   
# if field == 'rho': 
#     axs[0].plot(times, 
#             (coef1*np.cos(omegar1*times-phase1)*np.exp(gamma1*times))
#             ,label="exact1")
#     axs[0].plot(times, 
#             (coef*np.cos(omegar*times-phase)*np.exp(gamma*times))
#             + (coef1*np.cos(omegar1*times-phase1)*np.exp(gamma1*times))
#             ,label="exact01")
axs[0].set_title("Real part")
axs[1].set_title("Imaginary part")    
axs[2].set_title("Log plot of modulus squared")
fig.legend()
plt.show()
plt.savefig("FourierBiFi4Comp.png")
plt.close()


# %%
# Plot of field at dirrent time steps
plt.figure()
plt.plot(arr1[:,0], label ='initial')
plt.plot(arr1[:,1], label='step1')
plt.plot(arr1[:,15], label='step15')
plt.plot(arr1[:,50], label ='step50')
plt.xlabel('cell Z direction')
plt.title('Ez sum over x and y ');
plt.legend()
plt.show()
# print(arr1[:,1])
plt.savefig("fig2.png")
plt.close()

# %%
# read electric energy
tabE=pd.read_csv("ReducedDiagnostics/ElecEnergy.txt",delim_whitespace=True)
#tab.plot(1,2)
time = tabE.values[:,1]
ex2 = tabE.values[:,2]
ey2 = tabE.values[:,3]
ez2 = tabE.values[:,4]
etot = ex2+ey2+ez2
# read magentic energy
tabB=pd.read_csv("ReducedDiagnostics/MagEnergy.txt",delim_whitespace=True)
tB = tabB.values[:,1]
bx2 = tabB.values[:,2]
by2 = tabB.values[:,3]
bz2 = tabB.values[:,4]
btot = bx2+by2+bz2
# read particle kinetic energy
tabPart=pd.read_csv("ReducedDiagnostics/Part.txt",delim_whitespace=True)
tPart = tabPart.values[:,1]
ekin = tabPart.values[:,5]

# plots
plt.figure()
fig, axs = plt.subplots(2, 2,sharex=True,tight_layout=True)
axs[0,0].plot(time,ex2+ey2+ez2)
axs[0,0].set_title('electric energy')
axs[1,0].plot(time,bx2+by2+bz2)
axs[1,0].set_title('magnetic energy')
axs[0,1].plot(tPart,ekin)
axs[0,1].set_title('kinetic energy')
axs[1,1].plot(time,ex2+ey2+ez2+bx2+by2+bz2+ekin)
axs[1,1].set_title('total energy');
plt.show()
plt.savefig('fig3.png')
plt.close()

plt.figure()
plt.plot(time,ex2,label='Ex^2')
plt.plot(time,ey2,label='Ey^2')
plt.plot(time,ez2,label='Ez^2')
plt.legend()
plt.xlabel('t')
plt.show()
plt.savefig('ExEyEz.png')
plt.close
# %%
plt.figure()
plt.plot(time,ex2+ey2+ez2+bx2+by2+bz2+ekin)
plt.title('total energy');
plt.show()
plt.savefig('total_energy')
plt.close()

plt.figure()
# Log plot of electric energy 0.5 \int Ex**2 to check Landau damping
itmax=500*4;#arr.shape[1];#500
coef = 2 * epsilon * r
# Numerical
plt.plot(time[:itmax],np.log(ez2[:itmax]), label='Numerical')
# Analytical solution and linear damping rate
plt.plot(time[:itmax], np.log(L[0]*L[1]*L[2]*(coef*np.cos(omegar*time[:itmax]-phase)*np.exp(gamma*time[:itmax]))**2))
plt.plot(time[:itmax], np.log(L[0]*L[1]*L[2]*(coef*np.exp(gamma*time[:itmax]))**2))
plt.legend(loc='lower right')
plt.xlabel('t')
# plt.ylabel('Y axis label')
plt.show()
plt.savefig('landau_damping.png')
plt.close()
# %%
