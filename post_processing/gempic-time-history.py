# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
#

# %% [markdown]
# ## Time history diagnostics

# %%
# Import needed modules
import numpy as np
import matplotlib.pylab as plt
import os

# %%
# Set working directory
pathname = '/cobra/u/sonnen/gempic_runs/Weibel_long'
pathname_out = pathname + '/processed'
try:
    os.mkdir(pathname_out)
except(FileExistsError):
    pass
os.chdir(pathname)

# %%
# Read time history output file
sim_name = "Weibel"
with open(sim_name + '.txt', 'r') as file:
    lines = file.readlines()

nsteps = len(lines)-1  # First line is not counted
nsteps = 10000
print("nsteps=",nsteps)
time = np.zeros(nsteps)
ex = np.zeros(nsteps)
ey = np.zeros(nsteps)
ez = np.zeros(nsteps)
bx = np.zeros(nsteps)
by = np.zeros(nsteps)
bz = np.zeros(nsteps)
ekin = np.zeros(nsteps)
px = np.zeros(nsteps)
py = np.zeros(nsteps)
pz = np.zeros(nsteps)
errgauss = np.zeros(nsteps)
for i in range(nsteps):
    vals = lines[i+1].rstrip().split(' ')
    time[i] = vals[0]
    ex[i] = vals[1]
    ey[i] = vals[2]
    ez[i] = vals[3]
    bx[i] = vals[4]
    by[i] = vals[5]
    bz[i] = vals[6]
    ekin[i] = vals[7]
    px[i] = vals[8]
    py[i] = vals[9]
    pz[i] = vals[10]
    errgauss[i] = vals[11]
    #print(i,vals)

# %%
fig, axs = plt.subplots(2, 2,sharex=True,tight_layout=True)
axs[0,0].plot(time,ex+ey+ez)
axs[0,0].set_title('electric energy')
axs[1,0].plot(time,bx+by+bz)
axs[1,0].set_title('magnetic energy')
axs[0,1].plot(time,ekin)
axs[0,1].set_title('kinetic energy')
axs[1,1].plot(time,ex+ey+ez+bx+by+bz+ekin)
axs[1,1].set_title('total energy');

# %%
# Plot |Ex|**2 
fig, ax = plt.subplots()
ax.semilogy(time, ex, 'k')
ax.set_xlabel('time')
ax.set_ylabel('$E_x^2$')
ax.set_title('$E_x^2$ vs. Time')
ax.legend([sim_name],loc='upper left')
plt.savefig(pathname_out + '/E1_squared_vs_t.jpg');

# %%
# Plot Gauss Law Error 
fig, ax = plt.subplots()
ax.plot(time, errgauss);
ax.set_xlabel('time');
ax.set_ylabel('Gauss Law Error');
ax.set_title('Gauss Law Error vs. Time');
ax.legend([sim_name],loc='upper left')
plt.savefig(pathname_out + '/Gauss_Law_Error_vs_t.jpg');

# %%
# Comparison of |E1|**2, |E2|**2, |B3|**2
#cd(pathname);
fig, ax = plt.subplots()
ax.plot(time, np.log(ex));
ax.plot(time, np.log(ey));
ax.plot(time, np.log(bz));
ax.plot(time, 2*0.023*time -16)
#ax.set_ylim([-10,150]);
ax.set_xlabel('time');
ax.legend(['|E1|**2','|E2|**2','|B3|**2'],loc='upper left');
plt.savefig(pathname_out + '/E1_E2_B_squared_vs_t.jpg');

# %%
