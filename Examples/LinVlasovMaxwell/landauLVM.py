# %% [markdown]
# - Convert to jupyter notebook with `jupytext --to ipynb landauLVM.py`
# - back to python percent format with `jupytext --to py:percent --opt notebook_metadata_filter=-all landauLVM.ipynb`

#%%
import os
import numpy as np
import matplotlib.pyplot as plt

#%%
saveFigs = False

#%%
pathname = '.' # Notebook should be in directory where code is run
print('run directory: ', os.getcwd())
pathname_out = pathname + '/processed/'
try:
    os.mkdir(pathname_out)
except(FileExistsError):
    pass
os.chdir(pathname)

#%%
data_e = np.loadtxt("ReducedDiagnostics/ElecEnergy.txt", skiprows=1)
data_b = np.loadtxt("ReducedDiagnostics/MagEnergy.txt", skiprows=1)
data_p = np.loadtxt("ReducedDiagnostics/Part.txt", skiprows=1)

time = data_e[:,1]

energy_e = np.sum(data_e[:, 2:], axis=1)
energy_b = np.sum(data_b[:, 2:], axis=1)
energy_w = data_p[:,-2]
energy_p = data_p[:,-1]
energy_tot =  energy_e + energy_b + energy_w

# %% plot energies
plt.plot(time, energy_e, "-", label="Electric Energy")
plt.plot(time, energy_b, "-", label="Magnetic Energy")
plt.plot(time, energy_w, "-", label="Particle Energy Weights")
plt.plot(time, energy_tot, label="Total Energy")
plt.xlabel("$t$")
plt.legend()
if saveFigs: plt.savefig(pathname_out + "WeakLandauDamping_Energy.pdf", bbox_inches="tight")
plt.show()

# %%
plt.plot(time, (energy_tot-energy_tot[0])/energy_tot[0])
plt.title("Relative Error Total Energy")
plt.show()

# %%

# Parameters for exact solution
# least damped mode
epsilon = 0.04
r = 0.424666
phase = 0.3357725
coef = 2 * epsilon * r 
omegar = 1.2850
gamma = -0.0661
L = 2*np.pi/0.4

plt.plot(time, data_e[:,2], label="simulation")
plt.plot(time, L * (coef*np.cos(omegar*time-phase)*np.exp(gamma*time))**2, ls="--", label="analytical")
plt.yscale("log")
plt.ylabel("$|E_x|^2$")
plt.xlabel("$t$")
plt.legend()
if saveFigs: plt.savefig(pathname_out + "WeakLandauDamping.pdf", bbox_inches="tight")
plt.show()