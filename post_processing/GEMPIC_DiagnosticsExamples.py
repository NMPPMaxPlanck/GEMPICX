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

# %%
# Import needed modules (needs yt>=4 version, do pip install 'yt>=4.0' if needed)
# Classical yt plot examples can be found in Cookbook https://yt-project.org/doc/cookbook
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import yt
from yt.frontends.boxlib.data_structures import AMReXDataset
import pandas as pd

# Set working directory. Default is folder from which this notebook is run 
pathname = '/Users/sonnen/Codes/gempic_home/gempic/runs/Maxwell3D'
pathname_out = pathname + '/processed'
try:
    os.mkdir(pathname_out)
except(FileExistsError):
    pass
os.chdir(pathname)

# %% [markdown]
# ### Reduced diagnostics

# %%
# read electric energy
tabE=pd.read_csv("ReducedDiagnostics/ElecEnergy.txt",delim_whitespace=True)
#tab.plot(1,2)
min = 1
max =-1
time = tabE.values[min:max,1]
ex2 = tabE.values[min:max,2]
ey2 = tabE.values[min:max,3]
ez2 = tabE.values[min:max,4]
etot = ex2+ey2+ez2
# read magentic energy
tabB=pd.read_csv("ReducedDiagnostics/MagEnergy.txt",delim_whitespace=True)
tB = tabB.values[min:max,1]
bx2 = tabB.values[min:max,2]
by2 = tabB.values[min:max,3]
bz2 = tabB.values[min:max,4]
btot = bx2+by2+bz2
# read particle kinetic energy
tabPart=pd.read_csv("ReducedDiagnostics/Part.txt",delim_whitespace=True)
tPart = tabPart.values[min:max,1]
ekin = tabPart.values[min:max,5]

# plots
fig, axs = plt.subplots(2, 2,sharex=True,tight_layout=True)
axs[0,0].plot(time,np.log(ex2+ey2+ez2))
axs[0,0].set_title('electric energy')
axs[1,0].plot(time,np.log(bx2+by2+bz2))
axs[1,0].set_title('magnetic energy')
axs[0,1].plot(tPart,ekin)
axs[0,1].set_title('kinetic energy')
axs[1,1].plot(time,ex2+ey2+ez2+bx2+by2+bz2+ekin)
axs[1,1].set_title('total energy');

# %% [markdown]
# ### Full diagnostics: Fields 

# %%
# read times series of fields
ts = yt.load('./FullDiagnostics/plt_field??????')
# Print number of items in time series
ntz = ts.__len__()
print('number of time slices ', ntz)
# print field list of first time steps. They are the same for all time steps
print("yt field list ",ts[0].field_list)


# %%
# Process data from specific iteration
iter = 50
ds = ts[iter]
print(ds.field_list)
print(ts.params)

time = float(ds.current_time)
x_left = np.array(ds.domain_left_edge)
x_right = np.array(ds.domain_right_edge)
Ndim = ds.domain_dimensions
L = x_right - x_left
volume = L[0]*L[1]*L[2]

# %% [markdown]
# ### Example plots with yt

# %%
# Slice plot of B_z in z direction (with or without saving to a file)
#yt.SlicePlot(ds, 'z', "Bz").save()
yt.SlicePlot(ds, 'x', "Ex")

# %%
# Create a line plot of the variables 'Ex', 'Ey' and 'Bz' 
# spaced between the coordinates defined by the last two tuples
x_mid = 0.5*(x_right[0]-x_left[0])
y_mid = 0.5*(x_right[1]-x_left[1])
z_mid = 0.5*(x_right[2]-x_left[2])
print(x_left[0], y_mid, z_mid, x_left[1], y_mid, z_mid)
plot = yt.LinePlot(ds, [('boxlib', 'Ey'), ('boxlib', 'Bz'), ('boxlib', 'phi')], 
            (x_left[0], y_mid, z_mid), (x_right[0], y_mid, z_mid), Ndim[0])
# Add a legend
plot.annotate_legend(('boxlib', 'Ey'))

# Show or save the line plot
plot.show()
#plt.save()

# %%
plot = yt.LinePlot(ds, [('boxlib', 'rho')], (x_left[0], y_mid, z_mid), 
            (x_right[0], y_mid, z_mid), Ndim[0])
# Add a legend
plot.annotate_legend(('boxlib', 'rho'))

# show
plot.show()

# %% [markdown]
# ### Example plots with matplotlib 

# %%
# other field and particle quantities can be extracted an plotted with matplotlib
# Get field quantities
field_data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions )
#Ex = field_data['boxlib','Ex']
#rho = field_data['boxlib','rho']
Ey = field_data['boxlib','Ey']
#Ez = field_data['boxlib','Ez']
#Bx = field_data['boxlib','Bx']
#By = field_data['boxlib','By']
Bz = field_data['boxlib','Bz']
#Jx = field_data['boxlib','Jx']
#Jy = field_data['boxlib','Jy']
#Jz = field_data['boxlib','Jz']
extent = [ds.domain_left_edge[ds.dimensionality-1], ds.domain_right_edge[ds.dimensionality-1],
          ds.domain_left_edge[0], ds.domain_right_edge[0] ]
#plt.imshow(Bz[:,Bz.shape[1]//2,:], extent=extent, aspect='auto')
nx = Ey.shape[0]
ny = Ey.shape[1]
nz = Ey.shape[2]
x = np.linspace(x_left[0],x_right[0],nx)
plt.plot(x,Ey[:,0,0]-np.cos(x-time))
plt.plot(x,Bz[:,0,0]-np.cos(x-time))
#plt.plot(x,np.cos(x-time))
#plt.plot(x,np.cos(x-time))
#print("E_energy ", np.sum(Ex**2 + Ey**2 + Ez**2 )/(nx*ny*nz))
#print("B_energy ", np.sum(Bx**2 + By**2 + Bz**2)/(nx*ny*nz))
#print("E_energy ", np.sum(Ex**2 + Ey**2 )/(nx*ny*nz))
#print("Bz_energy ", np.sum(Bz**2)/(nx*ny*nz))

# %%
# Power spectrum in x,y - sum over z direction
Bzfft = np.fft.fftn(np.sum(Bz,2))
Bzfft2 = np.real(Bzfft * np.conj(Bzfft))

lvls = np.logspace(-6.0, 1, 20)
plt.cla()
Lxmax = int(L[0]/2)
Lymax = int(L[1]/2)
kx = 2*np.pi/L[0] * np.arange(Lxmax)
ky = 2*np.pi/L[1] * np.arange(Lymax)
plt.contourf(ky, ky, Bzfft2[0:Lxmax, 0:Lymax], cmap=cm.jet,norm=colors.LogNorm(), levels=lvls)
plt.colorbar()
#plt.xlabel('wave number' r'$\ k \Delta x$',fontsize=13)
plt.xlabel(r'$k_x$',fontsize=13)
plt.ylabel(r'$k_y$',fontsize=13)

# %% [markdown]
# ### Example particle plots

# %%
# read particle dataset 
iter = 0
plotfile = 'FullDiagnostics/{}{:06d}'.format('plt_part',iter)
pds = AMReXDataset(plotfile)
# Compute some needed quantities
x_left = np.array(ds.domain_left_edge)
x_right = np.array(ds.domain_right_edge)
Ndim = ds.domain_dimensions
L = x_right - x_left
volume = L[0]*L[1]*L[2]

# print field list 
print("yt field list ",pds.field_list)

# %% [markdown]
# ### Example particle plot with yt

# %%
# particle plot (only for positions)
pxy=yt.ParticlePlot(
    pds, 
    ("all", "particle_position_x"),
    ("all", "particle_position_y"),
    width=(0.5, 0.5)
)

pxy.show()
# particle plot colored by the particle weights
pxyw=yt.ParticlePlot(
    pds,
    ("all", "particle_position_x"),
    ("all", "particle_position_y"),
    [("all", "particle_weight")],
    width=(0.75, 0.75)  # particle size
)
pxyw.show()


# %% [markdown]
# ### Example particle plots with matplotlib

# %%
# Extract numpy arrays from yt dataset
ad = pds.all_data()
species = 'electrons'
x = ad[species, 'particle_position_x'].v
vx = ad[species, 'particle_vx'].v
vy = ad[species, 'particle_vy'].v
vz = ad[species, 'particle_vz'].v
w = ad[species, 'particle_weight'].v

# Plot
plt.scatter(x,vx,s=.01,c='k')

print("nb_part", x.size, 32*16*16)
print("kinetic energy", 0.5*np.sum(w*(vx**2+vy**2+vz**2))/volume)
print("vthx", np.sqrt(np.sum(w*(vx**2))/volume))
print("vthy", np.sqrt(np.sum(w*(vy**2))/volume))
print("vthz", np.sqrt(np.sum(w*(vz**2))/volume))

# %%
