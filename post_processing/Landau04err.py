import numpy as np
import sys

# evaluate errors of electric field, energy and Gauss
with open('Landau04.txt', 'r') as file:
    lines = file.readlines()

nsteps = len(lines)-1  # First line is not counted
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
err_gauss = np.zeros(nsteps)
tot_energy = np.zeros(nsteps)
err_energy = np.zeros(nsteps)
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
    err_gauss[i] = vals[11]
    tot_energy[i] = vals[12]
    err_energy[i] = vals[13]

toten = ex+ey+ez+bx+by+bz+ekin
if np.max(np.abs(err_gauss))> 1e-10:
    print("Gauss law error too large", np.max(np.abs(err_gauss)))
    sys.exit(1)

if np.max(np.abs(toten-toten[0]))> 1e-6:
    print("Summed energy error too large", np.max(np.abs(toten-toten[0])))
    sys.exit(1)

if np.max(np.abs(err_energy))> 1e-6:
    print("Output energy error too large", np.max(np.abs(err_energy)))
    sys.exit(1)

i1 = 100
i2 = 1000
# the analytical solution of the linear Landau damping problem for k=0.4
E1ex = 0.5*(0.05*np.pi*0.4246*np.exp(-0.0661*time[i1:i2])*np.cos(1.2850*time[i1:i2]-0.335))**2
if np.linalg.norm(ex[i1:i2]-E1ex)/np.sqrt(i2-i1) > 1.e-4:
    print("Ex error too large", np.linalg.norm(ex[i1:i2]-E1ex)/np.sqrt(i2-i1))
    sys.exit(1)
        
    
print("Job succeeded")  