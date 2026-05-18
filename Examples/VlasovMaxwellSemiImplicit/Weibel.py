# %% [markdown]
# - Convert to jupyter notebook with `jupytext --to ipynb Weibel.py`
# - back to python percent format with `jupytext --to py:percent --opt notebook_metadata_filter=-all Weibel.ipynb`

# %% [markdown]
# ## Time history diagnostics

# %%
# Import needed modules
import numpy as np
import matplotlib.pylab as plt
import os
import pandas as pd
from scipy.signal import find_peaks
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
field = 'Bz'
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

# plots 
# Parameters for exact solution
# least damped mode
epsilon = 0.04
r = 0.424666
phase = 0
coef = 2 * epsilon * r 
omegar = 0
gamma = 0.2258

fig, axs = plt.subplots(3, 1,sharex=True,tight_layout=True)
# Analytical solution of fundamental mode (real part and imaginary part)
if field == 'rho':
    axs[0].plot(times, 
            (coef*np.cos(omegar*times-phase)*np.exp(gamma*times))
            ,label="exact")
    axs[1].plot(times, np.zeros_like(times))
elif field == 'Bz':
    coef = 2 * epsilon * r    
    axs[0].plot(times, np.zeros_like(times))
    #axs[1].plot(times, (coef*np.cos(omegar*times-phase)*np.exp(gamma*times)),label="exact")
# Analytical solution of fundamental mode (mudulus squared)    
#axs[2].plot(times, np.log((coef*np.cos(omegar*times-phase)*np.exp(gamma*times))**2))

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
axs[2].set_ylim([-30,0])
fig.legend()
#plt.savefig("FourierBiFi4Comp")
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
# read magnetic energy
try:
    tabB=pd.read_csv("ReducedDiagnostics/MagEnergy.txt",sep=r'\s+')
    tB = tabB.values[:,1]
    bx2 = tabB.values[:,2]
    by2 = tabB.values[:,3]
    bz2 = tabB.values[:,4]
except:
    print('MagEnergy.txt not found')
    bx2 = np.zeros_like(time)
    by2 = np.zeros_like(time)
    bz2 = np.zeros_like(time)
btot = bx2+by2+bz2
# read particle diagnostics
try:
    tabPart=pd.read_csv("ReducedDiagnostics/Part.txt",sep=r'\s+')
    tPart = tabPart.values[:,1]
    px = tabPart.values[:,2]
    py = tabPart.values[:,3]
    pz = tabPart.values[:,4]
    # read kinetic energy
    ekin = tabPart.values[:,5]
except:
    print('Part.txt not found')

# read error on Gauss law
try:
    tabGauss=pd.read_csv("ReducedDiagnostics/GaussError.txt",sep=r'\s+')
    tGauss = tabGauss.values[:,1]
    gaussError = tabGauss.values[:,2]
except:
    print('Gauss.txt not found')

# plots
fig, axs = plt.subplots(3, 2,sharex=True,tight_layout=True)
axs[0,0].plot(time,ex2+ey2+ez2)
axs[0,0].set_title('electric energy')
axs[1,0].plot(time,bx2+by2+bz2)
axs[1,0].set_title('magnetic energy')
axs[0,1].plot(tPart,ekin)
axs[0,1].set_title('kinetic energy')
axs[1,1].plot(time,ex2+ey2+ez2+bx2+by2+bz2+ekin)
axs[1,1].set_title('total energy')
axs[2,0].plot(time,px)
axs[2,0].plot(time,py)
axs[2,0].plot(time,pz)
# label for particle momentum at the top
#axs[2,0].legend(['px','py','pz'])
axs[2,0].set_title('total particle momentum')
axs[2,1].plot(time,gaussError)
axs[2,1].set_title('Error on Gauss Law')
plt.show()

# %%
# Plot |Bz|**2 
fig, ax = plt.subplots()
ax.semilogy(time, bz2)
ax.semilogy(time, ex2)
ax.semilogy(time, ey2)
ax.semilogy(time, by2)
ax.semilogy(time,8e-10*np.exp(2*0.02784*time))
ax.set_xlabel('time')
ax.set_ylabel('$Bz^2$')
ax.set_ylim([1e-7,1e-1])
ax.legend([r'$\frac {1}{2}|B_z|^2$',r'$\frac {1}{2}|E_x|^2$',r'$\frac {1}{2}|E_y|^2$',r'$\frac {1}{2}|B_y|^2$'],loc='upper left')
#plt.savefig(pathname_out + '/B3_squared_vs_t.jpg')
plt.show()

# %%
i1 = 6000
i2 = 10000
plt.semilogy(time[i1:i2],by2[i1:i2])
plt.semilogy(time[i1:i2],bz2[i1:i2])
plt.semilogy(time[i1:i2],by2[i1:i2]+bz2[i1:i2])

imax, Eval_dict = find_peaks(bz2[i1:i2],height=0)
Eval = Eval_dict['peak_heights']
slope = np.zeros(Eval.shape)
freq = np.zeros(Eval.shape)
for i in range(Eval.size-1):
    slope[i] = (np.log(Eval[i+1])-np.log(Eval[i]))/(time[imax[i+1]]-time[imax[i]])
    freq[i] = np.pi / (time[imax[i+1]] - time[imax[i]])
slope_av = np.average(slope[:-1])
freq_av = np.average(freq[:-1])
print("growth rate", slope_av/2)
print("frequency", freq_av)
print(slope[:-1]/2)
print(2*freq[:-1])
#print(imax)
#print(by2[0],bz2[0])

# %%
# Plot |E2|**2, |B3|**2 actual and theoretical growth rate
fig, ax = plt.subplots()
i1 = 5000
i2 = 15000
ax.plot(time[i1:i2], np.log(ey2[i1:i2]))
ax.plot(time[i1:i2], np.log(by2[i1:i2]))
ax.plot(time[i1:i2], np.log(bz2[i1:i2]))
ax.plot(time[i1:i2], np.log(by2[i1:i2]+bz2[i1:i2]))
ax.plot(time[i1:i2], np.log(ex2[i1:i2]))
#ax.plot(time[i1:i2], 2*0.02784*time[i1:i2] - 18.1)   # theoretical growth rate
ax.plot(time[i1:i2], 2*0.0245*time[i1:i2] - 16.7)
ax.set_xlabel('time')
plt.show()

# %%