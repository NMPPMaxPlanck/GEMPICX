# %% [markdown]
# - Convert to jupyter notebook with `jupytext --to ipynb ColdPlasmaDispersion.py`
# - back to python percent format with `jupytext --to py:percent --opt notebook_metadata_filter=-all ColdPlasmaDispersion.ipynb`

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import yt
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
ntz = len(ts) # number of items in time series
# save times corresponding to each dataset
times = np.zeros((ntz),dtype=float) 
for i in range(ntz):
    times[i] = ts[i].current_time

# %%
ds = ts[-1]
time = ds.current_time
nx, ny, nz = ts[0].domain_dimensions
x_left = np.array(ds.domain_left_edge)
x_right = np.array(ds.domain_right_edge)
L = x_right - x_left

# %%
# Analytical dispersion relation
params = {'B0x': 1., 'B0y': 1., 'B0z': 1., 'n0': 1., 'alpha': 1., 'epsilon': 1.}

# Swap B0x and B0z since simulations is orientated differently to derivation of dispersion relation
params["B0x"], params["B0z"] = params["B0z"], params["B0x"]

# One complex array for each branch
kvec = np.linspace(0, 7, 100)
tmps = []
nbranches = 4
for n in range(nbranches):
    tmps += [np.zeros_like(kvec, dtype=complex)]


# angle between k and magnetic field
if params['B0z'] == 0:
    theta = np.pi/2
else:
    theta = np.arctan(
        np.sqrt(params['B0x']**2 + params['B0y']**2) / params['B0z'])
print(theta)
cos2 = np.cos(theta)**2

neq = params['n0']

# powers of parameters
B2 = params['B0x']**2 + \
    params['B0y']**2 + params['B0z']**2
alpha2 = params['alpha']**2
alpha4 = params['alpha']**4
alpha6 = params['alpha']**6
eps2 = params['epsilon']**2
eps4 = params['epsilon']**4
eps6 = params['epsilon']**6
k2vec = kvec**2

for n, k2 in enumerate(k2vec):
    # polynomial coefficients in order of increasing degree
    # 0th degree
    a = B2*k2**2*neq*alpha2*eps2*cos2
    # 1st degree in omega^2
    b = -neq**3 * alpha6 - B2*k2*neq*alpha2*eps2 - 2*k2*neq**2*alpha4*eps2 - \
        B2*k2*neq*alpha2*eps2*cos2 - B2*k2**2*eps4 - k2**2*neq*alpha2*eps4
    # 2nc degree in omega^2
    c = B2*neq*alpha2*eps2 + 3*neq**2*alpha4*eps2 + 2 * \
        B2*k2*eps4 + 4*k2*neq*alpha2*eps4 + k2**2*eps6
    # 3rd degree in omega^2
    d = -B2*eps4 - 3*neq*alpha2*eps4 - 2*k2*eps6
    # 4th degree in omega^2
    e = eps6

    # determinant in polynomial form
    det = np.polynomial.Polynomial([a, b, c, d, e])

    # solutions
    sol = np.sqrt(np.abs(det.roots()))
    # Ion-cyclotron branch
    tmps[0][n] = sol[0]
    # Electron-cyclotron branch
    tmps[1][n] = sol[1]
    # L-branch
    tmps[2][n] = sol[2]
    # R- branch
    tmps[3][n] = sol[3]

# %%
# mpirun -np X python ../SupplementaryScripts/PostProcessing/CreateSpaceTimeArrays.py Ez
arr = np.load("t_x_array.npy")    

# apply the hann filter
hann = np.hanning(ntz)
arr = arr * hann
# FFT in space and time
arrfft = np.fft.fftn(arr)
# normalize and transpose FFT array for plots
arrfftnorm = np.transpose(np.abs(arrfft))/np.abs(arrfft).max()
[N,L] = arrfftnorm.shape 

T = time # last current time
Lx = x_right[0] - x_left[0]

# define frequency (om) and wave number (kx) values
Lmax = int(L/2)
Nmax = int(N/2)
om = 2*np.pi/T * np.arange(Nmax)
kx = 2*np.pi/Lx * np.arange(Lmax)

lvls = np.logspace(-8, -0.5, 100)

plt.contourf(kx, om, arrfftnorm [0:Nmax, 0:Lmax], cmap="plasma",norm=colors.LogNorm(), levels=lvls, extend="both")
plt.xlim([0, 6])
plt.ylim([0, 5])
#plt.colorbar()

special = False
if special:
    # B = Bz -> combine two modes
    kvecc = np.hstack((kvec[np.real(tmps[1])<0.99999999], kvec[np.real(tmps[2])>1.00000001]))
    tmpss = np.hstack((tmps[1][np.real(tmps[1])<0.99999999], tmps[2][np.real(tmps[2])>1.00000001]))
    plt.plot(kvec, tmps[0], "--", c="tab:blue")
    plt.plot(kvecc, tmpss, "--", c="tab:green")
    plt.plot(kvec, tmps[3], "--", c="tab:red")
else:
    # Other cases
    for i in range(nbranches):
        plt.plot(kvec, tmps[i], "--")

#plt.xlabel('wave number' r'$\ k \Delta x$',fontsize=13)
plt.xlabel('wave number' r'$\ k$',fontsize=13)
plt.ylabel('frequency' r'$\ \omega$',fontsize=13)
plt.savefig(pathname_out + '/coldplasma_dispersion.png', dpi=300, bbox_inches="tight")
plt.show()
# %%
