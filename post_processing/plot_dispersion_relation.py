# %%
# # run this script in parallel with mpirun -n 4 python3 plot_dispersion_relation.py (for 4 cores)
# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import yt

# path to gempic directory
gempic_dir = '/Users/sonnen/Codes/gempic_home/gempic'
# Path to folder with simulation data
pathname = '/Users/sonnen/Codes/gempic_home/gempic_runs/BernsteinTestDiag' 
pathname_out = pathname + '/processed'
try:
    os.mkdir(pathname_out)
except(FileExistsError):
    pass
os.chdir(pathname)

# read times series
ts = yt.load('./Plotfiles/rho?????')
ntz = ts.__len__() # number of items in time series
ds=ts[-1]
# print field list and choose field to be used for dispersion relation
print(ds.field_list)

nx, ny, nz = ds.domain_dimensions
x_left = np.array(ds.domain_left_edge)
x_right = np.array(ds.domain_right_edge)
L = x_right - x_left
time = float(ds.current_time)
print(time)
# %%
arr=np.load("t_x_array.npy")
# apply the hann filter
hann = np.hanning(ntz);
arr = arr * hann
# FFT in space and time
arrfft = np.fft.fftn(arr)
# normalize and transpose FFT array for plots
arrfftnorm = np.transpose(np.abs(arrfft))/np.abs(arrfft).max()
[N,L] = arrfftnorm.shape 
# %%
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
#print("om ", Nmax, om)
# %%
from scipy.special import i0, iv  # modified bessel function
from scipy.optimize import brenth
# Bernstein wave dispersion function from Bernstein 1958 (Z is approximated)
# also Mario Raeth's thesis formula (4.1.4)
def D(om):
    s = np.exp(-k**2)*iv(0,k**2) - 2 # Gamma_0 - 2
    for n in range(1,50):
        s = s + om * np.exp(-k**2)*iv(n,k**2) * (1/(om - n) + 1/(om + n))
    return s
nom = int(om[-1])
roots = np.zeros((nom,Lmax))
for i in range(1,Lmax):
    k = kx[i-1]
    for j in range (nom):
        try:
            roots[j,i] = brenth(D,j+1.00000000000001,j+1.99999999999999)
        except(ValueError):
            roots[j,i] = 0

# %%
# contour plots of FFT array
lvls = np.logspace(-9.0, -1, 50)
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


# %%
