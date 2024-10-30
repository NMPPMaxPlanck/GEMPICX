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

# %%
# Set working directory
pathname = '.'
pathname_out = pathname + '/processed'
try:
    os.mkdir(pathname_out)
except(FileExistsError):
    pass
os.chdir(pathname)

# %%
# read times series
tsfield = yt.load('./FullDiagnostics/plt_field??????')

iteration = 2
ds = tsfield[iteration]
print(ds.field_list)

time = float(ds.current_time)
x_left = np.array(ds.domain_left_edge)
x_right = np.array(ds.domain_right_edge)
L = x_right - x_left
#print(step,time)
print(L[0],L[1],L[2])
volume = L[0]*L[1]*L[2]

# %%
# Create a line plot of the variables 'rho' and 'Ex' with 100 sampling points evenly
# spaced between the coordinates (0, 1, 1) and (L[0], 1, 1)
yt.LinePlot(ds, [('boxlib', 'rho'), ('boxlib', 'Ex')], (0, 1, 1), (L[0], 1, 1), 100)

# %%
yt.LinePlot(ds, [('boxlib', 'rho')], (0, 1, 1), (L[0], 1, 1), 100)

# %%
# Extract field as numpy array
field_data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions )
Ex = field_data['boxlib','Ex']
phi = field_data['boxlib','phi']
rho = field_data['boxlib','rho']
print('max: ',np.max(Ex), np.max(phi), np.max(rho))
plt.plot(Ex[:,0,0])

 # %%
 # Slice plot of E_x (with and without saving to a file)
yt.SlicePlot(ds, 'y', "Ex").save()
yt.SlicePlot(ds, 'z', "Ex")

# %%
# Read particle data. Needs AMReXDataset 
plotfile = "./FullDiagnostics/plt_part000000"
dsp = AMReXDataset(plotfile)
print(dsp.field_list)
# particle plot (x-y)
pxy=yt.ParticlePhasePlot(
    dsp,
    ("all", "particle_position_x"),
    ("all", "particle_position_y")
)
pxy.show()

# particle plot (x-vx)
pxvx=yt.ParticlePhasePlot(
    dsp,
    ("all", "particle_position_x"),
    ("all", "particle_vx")
)
pxvx.set_log(("all", "particle_vx"),False)
pxvx.show()
# particle plot (y-z) colored by the particle weights
pxyw=yt.ParticlePhasePlot(
    dsp,
    ("all", "particle_position_y"),
    ("all", "particle_position_z"),
    [("all", "particle_weight")]
)
pxyw.show()


# %%
# other field and particle quantities can be extracted an plotted with matplotlib
# Get field quantities
field_data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions )
Ex = field_data['boxlib','Ex']
Ey = field_data['boxlib','Ey']
Ez = field_data['boxlib','Ez']
Bx = field_data['boxlib','Bx']
By = field_data['boxlib','By']
Bz = field_data['boxlib','Bz']
#Jx = field_data['boxlib','Jx']
#Jy = field_data['boxlib','Jy']
#Jz = field_data['boxlib','Jz']
extent = [ds.domain_left_edge[ds.dimensionality-1], ds.domain_right_edge[ds.dimensionality-1],
          ds.domain_left_edge[0], ds.domain_right_edge[0] ]
#plt.imshow(Bz[:,Bz.shape[1]//2,:], extent=extent, aspect='auto')
nx = Ex.shape[0]
ny = Ex.shape[1]
nz = Ex.shape[2]
x = np.linspace(x_left[0],x_right[0],nx)
plt.plot(x,Bz[:,0,0])
print("E_energy ", np.sum(Ex**2 + Ey**2 + Ez**2 )/(nx*ny*nz))
print("B_energy ", np.sum(Bx**2 + By**2 + Bz**2)/(nx*ny*nz))
print("Bz_energy ", np.sum(Bz**2)/(nx*ny*nz))

# %%
# Power spectrum in x,y - sum over z direction
Exfft = np.fft.fftn(np.sum(Ex,2))
Exfft2 = np.real(Exfft * np.conj(Exfft))

lvls = np.logspace(-6.0, 1, 20)
plt.cla()
Lxmax = int(L[0]/4)
Lymax = int(L[1]/4)
kx = 2*np.pi/L[0] * np.arange(Lxmax)
ky = 2*np.pi/L[1] * np.arange(Lymax)
kX, kY = np.meshgrid(kx, ky)
plt.contourf(kX, kY, Exfft2[0:Lxmax, 0:Lymax], cmap=cm.jet,norm=colors.LogNorm(), levels=lvls)
plt.colorbar()
#plt.xlabel('wave number' r'$\ k \Delta x$',fontsize=13)
plt.xlabel(r'$k_x$',fontsize=13)
plt.ylabel(r'$k_y$',fontsize=13)
print(Exfft2)

# %%
# Read particle data. Needs AMReXDataset 
plotfile = "./fullDiagnostics/plt_part000000"
ds = AMReXDataset(plotfile)
print(ds.field_list)
# Get particle quantities
ad = ds.all_data()
species = 'electrons'
x = ad[species, 'particle_position_x'].v
y = ad[species, 'particle_position_y'].v
vx = ad[species, 'particle_vx'].v
vy = ad[species, 'particle_vy'].v
vz = ad[species, 'particle_vz'].v
w = ad[species, 'particle_weight'].v

# Plot
plt.scatter(x,vx,s=.01,c='k')
#plt.scatter(x,y,s=.01,c='k')

print("nb_part", x.size, 32*16*16)
print("kinetic energy", 0.5*np.sum(w*(vx**2+vy**2+vz**2))/volume)
print("vthx", np.sqrt(np.sum(w*(vx**2))/volume))
print("vthy", np.sqrt(np.sum(w*(vy**2))/volume))
print("vthz", np.sqrt(np.sum(w*(vz**2))/volume))

# %%
