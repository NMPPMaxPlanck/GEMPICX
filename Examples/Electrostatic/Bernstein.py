# %% [markdown]
# - Convert to jupyter notebook with `jupytext --to ipynb Bernstein.py`
# - back to python percent format with `jupytext --to py:percent --opt notebook_metadata_filter=-all Bernstein.ipynb`

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pandas as pd
from scipy.special import iv  # modified bessel function
from scipy.optimize import brenth
import yt
#from yt.frontends.boxlib.data_structures import AMReXDataset 
#yt.enable_parallelism()
yt.set_log_level(0) # do not show log output

# %%
pathname = '.' # Add the path to your data folder
pathname_out = pathname + '/processed'
try:
    os.mkdir(pathname_out)
except(FileExistsError):
    pass
os.chdir(pathname)
# read times series
ts = yt.load('./FullDiagnostics/plt_field??????')
ntz = len(ts) # number of items in time series

# print field list and choose field to be used for dispersion relation
print(ts[0].field_list)
ds = ts[-1]
time = ds.current_time

# %%
# get space dimensions
nx, ny, nz = ts[0].domain_dimensions
x_left = np.array(ds.domain_left_edge)
x_right = np.array(ds.domain_right_edge)
L = x_right - x_left

# In order to generale file t_x_array.npy containing the array to be Fourier transformed in x and time run (adpating the path to the gempic directory and the number of processes)
# mpirun -n 10 python3 ../SupplementaryScripts/PostProcessing/CreateSpaceTimeArrays.py FullDiagnostics/plt_field rho
arr = np.load("t_x_array.npy")    

# apply the hann filter
hann = np.hanning(ntz)
arr = arr * hann
# FFT in space and time
arrfft = np.fft.fftn(arr)
print("fft ",arrfft[0,0])
# normalize and transpose FFT array for plots
arrfftnorm = np.transpose(np.abs(arrfft))/np.abs(arrfft).max()
[N,L] = arrfftnorm.shape 

T = time # last current time
Lx = x_right[0] - x_left[0]
print(ntz,N,L,T,2*np.pi/Lx,2*np.pi/T)

# define frequency (om) and wave number (kx) values
Lmax = int(L/2)
Nmax = int(N/2)
om = 2*np.pi/T * np.arange(Nmax)
kx = 2*np.pi/Lx * np.arange(Lmax)
# subrange for plotting
Lmax = int(L/4)
Nmax = int(N/32)
om = 2*np.pi/T * np.arange(Nmax)
kx = 2*np.pi/Lx * np.arange(Lmax)
kmax= int(Lmax/4)
omax = int(Nmax/4)
#print(om)

# %%
# Bernstein wave dispersion function from Bernstein 1958 (Z is approximated)
# also Mario Raeth's thesis formula (4.1.4) for Te = 1
Te = 2.5
def D(om):
    s = (np.exp(-k**2)*iv(0,k**2) - 1)/Te -1
    for n in range(1,50):
        s = s + om * np.exp(-k**2)*iv(n,k**2) *(1/(om - n) + 1/(om+n))
    return s
nom = int(om[-1])
roots = np.zeros((nom,Lmax))
for i in range(1,Lmax):
    k = kx[i-1]
    for j in range (nom):
        try:
            eps = 1.e-12
            roots[j,i] = brenth(D,j+1+eps,j+2-eps)
        except(ValueError):
            roots[j,i] = 0

# %%
lvls = np.logspace(-9., -2, 100)
#lvls = np.logspace(-8.0, 3, 20)
plt.cla()

#plt.contourf(kx[0:kmax], om[0:omax], a[0:kmax, 0:omax], cmap=cm.jet,norm=colors.LogNorm(), levels=lvls)
plt.contourf(kx[1:], om, arrfftnorm [0:Nmax, 1:Lmax], cmap=cm.jet,norm=colors.LogNorm(), levels=lvls)
plt.colorbar()
#plt.xlabel('wave number' r'$\ k \Delta x$',fontsize=13)
plt.xlabel('wave number' r'$\ k$',fontsize=13)
plt.ylabel('frequency' r'$ \ \omega$',fontsize=14)
# plot analytical solutions
for i in range(nom):
    plt.plot(kx,roots[i,:],'k.',markersize=2)

plt.savefig(pathname_out+'/dispersion_relation')
plt.show()


# %%
# read electric energy
tabE=pd.read_csv("ReducedDiagnostics/ElecFieldEnergy.txt",delim_whitespace=True)
#tab.plot(1,2)
time = tabE.values[:,1]
ex2 = tabE.values[:,2]
ey2 = tabE.values[:,3]
ez2 = tabE.values[:,4]
etot = ex2+ey2+ez2
# read magentic energy
tabB=pd.read_csv("ReducedDiagnostics/MagFieldEnergy.txt",delim_whitespace=True)
tB = tabB.values[:,1]
bx2 = tabB.values[:,2]
by2 = tabB.values[:,3]
bz2 = tabB.values[:,4]
btot = bx2+by2+bz2
# read particle kinetic energy
tabPart=pd.read_csv("ReducedDiagnostics/Particle.txt",delim_whitespace=True)
tPart = tabPart.values[:,1]
ekin = tabPart.values[:,5]

# plots
fig, axs = plt.subplots(2, 2,sharex=True,tight_layout=True)
axs[0,0].plot(time,ex2+ey2+ez2)
axs[0,0].set_title('electric energy')
axs[1,0].plot(time,bx2+by2+bz2)
axs[1,0].set_title('magnetic energy')
axs[0,1].plot(tPart,ekin)
axs[0,1].set_title('kinetic energy')
axs[1,1].plot(time,ex2+ey2+ez2+bx2+by2+bz2+ekin)
axs[1,1].set_title('total energy')
plt.show()


# %%
