# %% [markdown]
# - Convert to jupyter notebook with `jupytext --to ipynb TwoStream.py`
# - back to python percent format with `jupytext --to py:percent --opt notebook_metadata_filter=-all TwoStream.ipynb`

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
# $$ \hat{E}(k,t) = 2 \epsilon r \exp(\gamma t) \cos(\omega_r t - \varphi )$$ 
# - $\epsilon$ is the initial perturbation of the density given in the input file
# - $r$, $\gamma$, $\omega_r$ and $\varphi$ are obtained from the dispersion solver
# - Here the values have been computed for $-v_1=v_2=1.3$ and  $-v_1=v_2=2.4$
# - other values can be obtained with the dispersion relation solver        

# %%

# plots 
# Parameters for exact solution
v2 = 1.3
# 1) For v1 = -1.3 and v2 = 1.3 
if v2 == 1.3:
    eps = 0.01
    r = 0.96334652 #0.90513422
    coef = 2 * eps * r
    gamma = -0.0010397510581334906
    omegar =  1.1648636041653582
    phi = -6.2643199
    timesE = times # time for exact solution only needed until saturation
# 2) For v1 = -2.4 and v2 = 2.4
if v2 == 2.4:
    eps = 0.001
    r = 0.22332107 # 6.27853332e-01
    coef = 2 * eps * r
    gamma = 0.22584425503471
    omegar = 0
    phi = 0
    timesE = times[:1100] # time for exact solution only needed until saturation

fig, axs = plt.subplots(3, 1,sharex=True,tight_layout=True)
# Analytical solution of fundamental mode (real part and imaginary part)
if field == 'rho':
    axs[0].plot(times, 
            (coef*np.cos(omegar*times-phi)*np.exp(gamma*timesE))
            ,label="exact")
    axs[1].plot(times, np.zeros_like(times))
elif field == 'Ex':   
    axs[0].plot(times, np.zeros_like(times))
    axs[1].plot(timesE, (coef*np.cos(omegar*timesE-phi)*np.exp(gamma*timesE)),label="exact")
# Analytical solution of fundamental mode (modulus squared)   
axs[2].plot(timesE, np.log((coef*np.cos(omegar*timesE-phi)*np.exp(gamma*timesE))**2))

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

# %% [markdown]
# ### Time evolution of reduced diagnostics 
# - Electric energy
# - Particle kinetic energy
# - Total momentum of particles
# - Total energy

# %%
# read electric energy
tabE=pd.read_csv("ReducedDiagnostics/ElecEnergy.txt",sep=r'\s+')

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
axs[1,0].plot(time,py)
axs[1,0].plot(time,pz)
axs[1,0].set_title('total momentum')
axs[0,1].plot(tPart,ekin)
axs[0,1].set_title('kinetic energy')
axs[1,1].plot(time,(ex2+ey2+ez2+ekin)/total_energy0)
axs[1,1].set_title('total energy')
print('Error in energy conservation ',np.abs(1-np.min((ex2+ey2+ez2+ekin)/total_energy0)))
#plt.savefig("TSCons.pdf")
plt.show()



# %%
# Plot of field at different time steps
plt.plot(arr1[:,0])
plt.plot(arr1[:,1])
plt.plot(arr1[:,15])
plt.plot(arr1[:,1400])
plt.show()

# %%
