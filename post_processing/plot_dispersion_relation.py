# run this script in parallel with mpirun -n 4 python3 plot_dispersion_relation.py (for 4 cores)

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import mpi4py
import yt

yt.enable_parallelism()
yt.set_log_level(0) # do not show log output

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()

pathname = '/Users/sonnen/Codes/gempic_home/gempic_runs/DispPar128' # Path to your data folder
pathname_out = pathname + '/processed'
try:
    os.mkdir(pathname_out)
except(FileExistsError):
    pass
os.chdir(pathname)
sim_name = "DispPerpB"

# read times series
ts = yt.load('./Plotfiles/' + sim_name + '*')
ntz = ts.__len__() # number of items in time series

# print field list and choose field to be used for dispersion relation
if rank == 0:
    print(ts[0].field_list)
field = 'E_y'    
# read in the data for each time slice
storage = {}
for store, ds in ts.piter(storage=storage):
    ad=ds.all_data()
    data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions )
    arr = np.array(data['boxlib',field])
    store.result = np.sum(np.sum(arr,2),1); # we sum over the y and z components
    time = float(ds.current_time)

# plots on rank 0 process
if rank == 0:
    # get space dimensions
    nx, ny, nz = ds.domain_dimensions
    x_left = np.array(ds.domain_left_edge)
    x_right = np.array(ds.domain_right_edge)
    L = x_right - x_left
    # fill array for FFT
    arr = np.zeros([nx,ntz]);
    for data in storage.items():
        arr[:,data[0]] = data[1]
    
    # apply the hann filter
    hann = np.hanning(ntz);
    arr = arr * hann
    # FFT in space and time
    arrfft = np.fft.fftn(arr)
    # normalize and transpose FFT array for plots
    arrfftnorm = np.transpose(np.abs(arrfft))/np.abs(arrfft).max()
    [N,L] = a.shape 
    
    T = time # last current time
    Lx = x_right[0] - x_left[0]
    print(ntz,N,L,T,2*np.pi/Lx,2*np.pi/T)

    # define frequency (om) and wave number (kx) values
    Lmax = int(L/2)
    Nmax = int(N/2)
    om = 2*np.pi/T * np.arange(Nmax)
    kx = 2*np.pi/Lx * np.arange(Lmax)

    # contour plots of FFT array
    plt.cla()
    lvls = np.logspace(-5.5, 0, 20) # fix level sets
    plt.contourf(kx, om, arrfftnorm[0:Nmax, 0:Lmax], cmap=cm.jet,norm=colors.LogNorm(), levels=lvls)
    plt.colorbar()
    plt.xlabel('wave number' r'$\ k$',fontsize=13)
    plt.ylabel('frequency' r'$ \ \omega$',fontsize=14)
    
    # Set plasma frequency and cyclotron frequency for analytical plots
    omp = 1
    omc = 1/2
    # plot analytical L mode
    omega=np.linspace(0.781,6,100)
    plt.plot(np.sqrt(omega**2 - omp**2 * (omega/(omega+omc))),omega)
    # plot analytical R mode (lower and upper branches)
    omega=np.linspace(0,omc-0.02,100)
    plt.plot(np.sqrt(omega**2 - omp**2 * (omega/(omega-omc))),omega)
    omega=np.linspace(1.2808,6,100)
    plt.plot(np.sqrt(omega**2 - omp**2 * (omega/(omega-omc))),omega)
    
    # save figure in a file
    plt.savefig(pathname_out+'/dispersion_relation')


