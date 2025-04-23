# %% [markdown]
# - Convert to jupyter notebook with `jupytext --to ipynb BumpOnTail.py`
# - back to python percent format with `jupytext --to py:percent --opt notebook_metadata_filter=-all BumpOnTail.ipynb`

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pandas as pd
import yt
yt.set_log_level(0) # do not show log output

# %%
pathname = '.' # Notebook should be in directory where code is run
pathname_out = pathname + '/processed'
try:
    os.mkdir(pathname_out)
except(FileExistsError):
    pass
os.chdir(pathname)
print('run directory: ', os.getcwd())
# read times series
ts = yt.load('./FullDiagnostics/plt_field??????')
ntz = len(ts) # number of items in time series
# save times corresponding to each dataset
times = np.zeros((ntz),dtype=float) 
for i in range(ntz):
    times[i] = ts[i].current_time

# print field list and choose field to be used for dispersion relation
ds = ts[0]
print(ds.field_list,ds.domain_left_edge, ds.domain_right_edge, ds.domain_dimensions)


# %%
# read in the data for each time slice
field = 'Ex'
storage = {}
for store, ds in ts.piter(storage=storage):
    data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions )
    arr = np.array(data['boxlib',field])
    store.result = np.sum(np.sum(arr,2),1) # we sum over the y and z components
    time = float(ds.current_time)

# %%
# get space dimensions
nx, ny, nz = ts[0].domain_dimensions
x_left = np.array(ds.domain_left_edge)
x_right = np.array(ds.domain_right_edge)
L = x_right - x_left
# fill array for FFT
arr = np.zeros([nx,ntz])
print(arr.shape)
for data in storage.items():
    arr[:,data[0]] = data[1] / (ny*nz)   

arr1 = arr
# FFT in space
arrfft = np.zeros_like(arr,dtype=complex)
for i in range(ntz):
    arrfft[:,i] = np.fft.fft(arr[:,i])
# Shift and normalize array
np.fft.fftshift(arrfft)
arrfft = arrfft / nx

# %% [markdown]
# ### Plots
# - plot first three Fourier modes. 
#    - Only first mode is excited. 
#    - The other give an idea of the noise and the nonlinear saturation 
# - The exact solution is for the 1D problem, but simulation can be run in 1D, 2D and 3D giving the same results  
# - Exact solution is computed with a Laplace transform in time and a Fourier transform in space. Only dominant first mode is computed and computed in the form
# $$ \hat{E}(k,t) = \epsilon r \exp(\gamma t) \exp(-i(\omega_r t - \varphi ))$$ 
# - $\epsilon$ is the initial perturbation of the density given in the input file
# - $r$, $\gamma$, $\omega_r$ and $\varphi$ are obtained from the dispersion solver 

# %%

# Parameters for exact solution
# least damped mode
epsilon = 0.001
r = 0.24190827
phase = -5.844418
coef = 4 * epsilon * r # not clear where factor 4 comes from, but it seems to match with numerical solution
omegar = 1.0012178936311045
gamma = 0.1980979758430256
timesE = times[0:1500] # time for exact solution only needed until saturation

fig, axs = plt.subplots(3, 1,sharex=True,tight_layout=True)
# Analytical solution of fundamental mode (real part and imaginary part)
if field == 'rho':
    axs[0].plot(times, 
            (coef*np.cos(omegar*times-phase)*np.exp(gamma*times))
            ,label="exact")
    axs[1].plot(times, np.zeros_like(times))
elif field == 'Ex': 
    axs[0].plot(timesE, -(coef*np.cos(omegar*timesE-phase)*np.exp(gamma*timesE)))
    axs[1].plot(timesE, (coef*np.sin(omegar*timesE-phase)*np.exp(gamma*timesE)),label="exact")
# Analytical solution of fundamental mode (mudulus squared)    
axs[2].plot(timesE, np.log((coef*np.exp(gamma*timesE))**2))

# Numerical solution of fundamental mode and harmonics to check noise level
for i in range(1,4):
    axs[0].plot(times, arrfft[i,:].real,label="k="+str(i))
    axs[1].plot(times, arrfft[i,:].imag)
    axs[2].plot(times, 2*np.log(np.abs(arrfft[i,:])))

axs[0].set_title("Real part")
axs[1].set_title("Imaginary part")    
axs[2].set_title("Log plot of modulus squared")
fig.legend()
plt.show()


# %%
# read electric energy
tabE=pd.read_csv("ReducedDiagnostics/ElecEnergy.txt",sep=r'\s+')
#tab.plot(1,2)
time = tabE.values[:,1]
ex2 = tabE.values[:,2]
ey2 = tabE.values[:,3]
ez2 = tabE.values[:,4]
etot = ex2+ey2+ez2
# read particle kinetic energy
tabPart=pd.read_csv("ReducedDiagnostics/Part.txt",sep=r'\s+')
tPart = tabPart.values[:,1]
ekin = tabPart.values[:,5]
# read particle kinetic momentum
px = tabPart.values[:,2]
py = tabPart.values[:,3]
pz = tabPart.values[:,4]

# plots
total_energy0 = ex2[0]+ey2[0]+ez2[0]+ekin[0]
fig, axs = plt.subplots(2, 2,sharex=True,tight_layout=True)
axs[0,0].plot(time,ex2+ey2+ez2)
axs[0,0].set_title('electric energy')
axs[1,0].plot(time,px)
#axs[1,0].plot(time,py)
#axs[1,0].plot(time,pz)
axs[1,0].set_title('momentum')
axs[0,1].plot(tPart,ekin)
axs[0,1].set_title('kinetic energy')
axs[1,1].plot(time,(ex2+ey2+ez2+ekin)/total_energy0)
axs[1,1].set_title('total energy')
print('Error in energy conservation ',np.abs(1-np.min((ex2+ey2+ez2+ekin)/total_energy0)))
plt.show()



# %%
# Log plot of electric energy 0.5 \int Ex**2
itmax=2000
coef = 4 * epsilon * r
#coef = .001
# Analytical solution and linear growth rate
plt.plot(time[:itmax], np.log(L[0]*L[1]*L[2]*(coef*np.exp(gamma*time[:itmax]))**2))
# Numerical
plt.plot(time[:itmax],np.log(ex2[:itmax]))
plt.show()


# %%
# Plot of field at dirrent time steps
plt.plot(arr1[:,0])
plt.plot(arr1[:,1])
plt.plot(arr1[:,15])
plt.plot(arr1[:,50])
plt.show()

# %%
