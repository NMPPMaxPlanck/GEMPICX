# %% [markdown]
# - Convert to jupyter notebook with `jupytext --to ipynb HPDRparallel.py`
# - back to python percent format with `jupytext --to py:percent --opt notebook_metadata_filter=-all HPDRparallel.ipynb`

# %%
GEMPICX_DIR = '../..' # user provided
import sys
sys.path.append(GEMPICX_DIR + '/Examples/SupplementaryScripts/DispersionRelations')
from zafpy import *
import numpy as np
import matplotlib.pyplot as plt
import cmath

# %% [markdown]
# ## Computes the dispersion relation for a hot plasma in the direction parallel to the magnetic field
# - These waves are Landau damped. The dispersion relation involves the plasma dispersion function Z
# - The least damped solution who often lies away from the other is computed with a Newton algorithm
# - The clustered damped roots with highest (negative) imaginary parts are computed with a contour integral method, implemented in zafpy.py 
#
# ### Warning
# - The code is not robust for low values of $k v_{th}$. Solutions can still be found by increasing the tolerance for the Newton solver or get_zeros.
# - This is due to the very strong variations of Z and the accumulation of round-off errors.
# - For such small values of $v_{th}$ the cold plasma dispersion relation should be used

# %%
# Define the physical parameters for the dispersion relation which will be used in the simulation
simulation = ['Vlasov-Maxwell', 'two species'] # only electrons: 'one species', electrons and ions: 'two species'

if simulation[1] == 'one species':
    mi = 1.0e16 # mass is set very large
else:
    mi = 4  # actual value for two species
me = 1
B0 = 1  # magnetic field strength in x or z direction
c = 1 # speed of light
vthe = .5  # thermal velocity of electrons. See warning about k*v_{th} above
vthi = vthe * np.sqrt(me / mi)
# Calculate the plasma frequencies
wce = B0 / me  # assuming e = 1
wpe = 1 / np.sqrt(me) # assuming a unit density 
wci = B0 / mi
wpi = wpe / np.sqrt(mi)
wp = np.sqrt(wpi**2 + wpe**2)
print("wpe = ", wpe, "wpi = ", wpi)
print("wce = ", wce, "wci = ", wci)
print("vthe = ", vthe, "vthi = ", vthi)




# %% [markdown]
# ## Longitudinal mode

# %%
# Dispersion relation in symbolic variables for the longitudinal mode
if simulation[1] == 'two species':
    def D(omega,k):
        return 1 + (wpe/(k*vthe))**2*(1+ (omega/(k*vthe*sp.sqrt(2))) *
        Z(omega/(k*vthe*sp.sqrt(2)))) + (wpi/(k*vthi))**2*(1+ (omega/(k*vthi*sp.sqrt(2))) *
        Z(omega/(k*vthi*sp.sqrt(2)))) 
else:
    def D(omega,k):
        return 1 + (wpe/(k*vthe))**2*(1+ (omega/(k*vthe*sp.sqrt(2))) *
        Z(omega/(k*vthe*sp.sqrt(2))))

# %% [markdown]
# - In order to find the roots of the dispersion function for given k, which is a complex function we plot it with mpmath.cplot
# - the complex argument (phase) is shown as color (hue) and the magnitude is show as brightness.
#   - This means that colors change quickly around a zero, which can be recognized as a black point with changing colors around 
#   - the white high brightness regions correspond to very large values, where contour integrals are hard to approximate numerically and so should be avoided

# %%
kmode = 1  # mode number
zaf=zafpy(D,kmode)

# set the range for the plot. This needs to be adjusted for the specific mode and parameters.
xmin= -2
xmax= 2
ymin = -2 
ymax = .1

fig = plt.figure()
ax = plt.subplot(111)
mp.cplot(zaf.D, re=[xmin, xmax], im=[ymin, ymax], points=7000, axes=ax)
ax.grid()
ax.set_xticks(np.linspace(xmin, xmax, 10))
ax.set_yticks(np.linspace(ymin, ymax, 9))
aspect_ratio = (xmax - xmin) / (ymax - ymin)
if aspect_ratio > 5:
    ax.set_aspect(0.2 * (xmax - xmin) / (ymax - ymin))
print("xmin = ", xmin, " xmax = ", xmax, " ymin = ", ymin, " ymax = ", ymax)
print("aspect ratio = ", (xmax - xmin) / (ymax - ymin))

# # Set boxes where zeros are expected
# points of boxes need to be in direct order: 
# z0 lower right, z1 upper right, z2 upper left, z3 lower left
# # 1) Least damped mode
fig = plt.figure()
ax = plt.subplot(111)
z0 = wp + (1.5 - 1.2j) * vthe * kmode
z1 = wp + (1.5 + 0.02j) * vthe * kmode
z2 = wp + (-0.5 + 0.02j) * vthe * kmode
z3 = wp + (-0.5 - 1.2j) * vthe * kmode
# Complex plot of the dispersion function and the box containing the least damped mode
mp.cplot(zaf.D, re=[xmin, xmax], im=[ymin, ymax], points=7000, axes=ax)
ax.plot([z0.real,z1.real],[z0.imag,z1.imag], 'k')
ax.plot([z1.real,z2.real],[z1.imag,z2.imag], 'k')
ax.plot([z2.real,z3.real],[z2.imag,z3.imag], 'k')
ax.plot([z3.real,z0.real],[z3.imag,z0.imag], 'k')

# 2) Higher order damped modes
fig = plt.figure()
ax = plt.subplot(111)
z0 = (4 - 3j) * vthe * kmode
z1 = (4 - 1.5j) * vthe * kmode
z2 = (3 - 1.5j) * vthe * kmode
z3 = (3 - 3j) * vthe * kmode
xmin = -6 * vthe * kmode
xmax = 6 * vthe * kmode
ymin = -4 * vthe * kmode
ymax = 0.1*vthe * kmode
# Complex plot of the dispersion function and the box containing the higher order damped mode
mp.cplot(zaf.D, re=[xmin, xmax], im=[ymin, ymax], points=7000, axes=ax)
ax.plot([z0.real,z1.real],[z0.imag,z1.imag], 'k')
ax.plot([z1.real,z2.real],[z1.imag,z2.imag], 'k')
ax.plot([z2.real,z3.real],[z2.imag,z3.imag], 'k')
ax.plot([z3.real,z0.real],[z3.imag,z0.imag], 'k')

print("Dispersion function for k = ", kmode )


# %%
# Compute least damped mode with Newton's method
z_least = zaf.compute_zeros_newton(wp, tol=1e-6, maxiter=1000, verbose=0)
print("Least damped mode: ", z_least)
print("D at least damped mode: ", zaf.D(z_least))


# %%
# Compute higher order zeros in box
print('number of zeros in box', zaf.count_zeros(z0, z1, z2, z3))
zaf.zeros=[]
zeros=zaf.get_zeros(z0, z1, z2, z3, tol = 1.e-6)
zeros.append(z_least)  # add least damped mode to list of zeros
zero_max=zeros[np.argmax(np.imag(zeros))]
zeros = sorted(zeros, key=lambda z: np.imag(z))
print("Zeros of D in box")
for z in zeros:
    print(z)
    if mp.fabs(zaf.D(z)) > 1e-6:
        print("Value of D at zero ",zaf.D(z))
print('------------------------')
print('k=',kmode)
print('zero with largest imaginary part (omega_j):', zero_max)

# %%
# Find the 4 least damped zeros for different k values
nk = 40
kk = np.linspace(0.1, 4, nk)
zeros_array = np.zeros((nk,2), dtype=complex)
z_least = wp # initial guess for least damped mode
for i in range(nk):
    kmode = kk[i]
    zaf = zafpy(D, kmode)
    # Compute least damped mode with Newton's method, using the last value as initial guess
    z_least = zaf.compute_zeros_newton(z_least, tol=1e-6)
    print('------------------------')
    print('k=',kmode)
    # Choose box for higher order zeros
    z0 = (4 - 3j) * vthe * kmode
    z1 = (4 - 1.5j) * vthe * kmode
    z2 = (3 - 1.5j) * vthe * kmode
    z3 = (3 - 3j) * vthe * kmode
    print('number of zeros in box', zaf.count_zeros(z0, z1, z2, z3))
    zaf.zeros=[]
    zeros=zaf.get_zeros(z0, z1, z2, z3, tol = 1.e-6)
    zeros.append(z_least)  # add least damped mode to list of zeros
    zeros = sorted(zeros, key=lambda z: -np.imag(z))
    zeros_array[i,:] = zeros[0:2] # store the 3 least damped zeros
    for z in zeros_array[i,:]:
        print(z)
    for z in zeros:
        print(z)

 

# %%
fig = plt.figure()
ax = plt.subplot(111)
mp.cplot(zaf.D, re=[z3.real,z0.real], im=[z0.imag,z1.imag], points=5000, axes=ax)
print("Zeros of D in box")
for z in zeros:
    print(z)
    if mp.fabs(zaf.D(z)) > 1e-8:
        print("Value of D at zero ",zaf.D(z))
zero_max= zeros[np.argmax(np.imag(zeros))]

# %%
# Plot real and imaginary parts of zeros for different k
plt.figure()
plt.plot(kk, np.real(zeros_array[:,1]), '.-', label='High order mode')
plt.plot(kk, np.real(zeros_array[:,0]), '.-', label='least damped zero')
plt.plot(kk, np.sqrt(wp**2 + 3 * vthe**2 * kk**2), '+', label='warm plasma')
plt.xlabel('k')
plt.ylabel('Real part of zero')
plt.legend()
plt.grid()
plt.title('Real part of zeros of D for different k')
plt.show()

plt.figure()
plt.plot(kk, np.imag(zeros_array[:,1]), '.-', label='High order mode')
plt.plot(kk, np.imag(zeros_array[:,0]), '.-', label='least damped zero')
plt.xlabel('k')
plt.xlabel('k')
plt.ylabel('Imaginary part of zero')
plt.legend()
plt.grid()
plt.title('Imaginary part of zeros of D for different k')
plt.show()

# %% [markdown]
# ## Transverse modes
# The dispersion relation can be separated into two parts one with left-handed polarization (L mode) and one with right-handed polarisation (R-mode). For only electrons they have respectively one negative root and two positive root and the other way round. Alltogether there are 6 roots, 3 positive and their 3 opposites. So that we can compute all the positive roots by considering only the L-mode dispersion relation and taking the opposite of the negative root.
# In the two species case there are two negative and two positive roots (with our choice of sign, the L-mode and ion cyclontron mode avec negative frequencies).
#
# There are also strongly damped roots, called high-order modes, the trace of which can be seen in the numerical dispersion relation. For this reason we also computed the least damped of these.

# %%
# Dispersion relation in symbolic variables
# Dispersion relation from Fitzpatrick (https://farside.ph.utexas.edu/teaching/plasma/Plasmahtml/node95.html) eq 7.98. Note that in his convention wce = -B0/me and wci = B0/mi, whereas for us both frequencies are positive.
if simulation[1] == 'two species':
    def D(omega,k):
        return (1 - (k*c)**2/omega**2 + wpe**2/(sp.sqrt(2)*omega*vthe*k)* Z((omega-wce)/(k*vthe*sp.sqrt(2)))
          + wpi**2/(sp.sqrt(2)*omega*vthi*k)* Z((omega+wci)/(k*vthi*sp.sqrt(2))))   
else:
    def D(omega,k):
        return  1 - (k*c)**2/omega**2 + wpe**2/(sp.sqrt(2)*omega*vthe*k) * Z((omega-wce)/(k*vthe*sp.sqrt(2)))

# %%
kmode = 0.5
zaf=zafpy(D,kmode)

xmin= -wp - c*2*kmode**2
xmax= wp + wce + c*2*kmode**2
ymin = -4 * vthe * kmode
ymax = 1 * vthe * kmode
plt.figure()
ax = plt.subplot(111)
mp.cplot(zaf.D, re=[xmin, xmax], im=[ymin, ymax], points=7000, axes=ax)
aspect_ratio = (xmax - xmin) / (ymax - ymin)
if aspect_ratio > 5:
    ax.set_aspect(0.2 * (xmax - xmin) / (ymax - ymin))
ax.grid()
ax.set_xticks(np.linspace(xmin, xmax, 10))
ax.set_yticks(np.linspace(ymin, ymax, 10))
ax.set_title('Dispersion function for k = ' + str(kmode))
plt.show()
print("Dispersion function for k = ", kmode )

# %%
# Compute zeros of the dispersion function in a box determined here
# Choose box where zeros are searched for (need to be positively oriented) 
# depending on the desired mode. See plot above to choose box.
# points of boxes need to be in direct order:
z0 = wce + (1. - 2j)*vthe*kmode # lower right corner
z1 = wce + (1. + 0.01j)*vthe*kmode # upper right corner
z2 = (0.01 + 0.01j)*vthe*kmode # upper left corner
z3 = (0.01 - 2j)*vthe*kmode # lower left corner

fig, ax = plt.subplots()
mp.cplot(zaf.D, re=[xmin, xmax], im=[ymin, ymax], points=7000, axes=ax)
#mp.cplot(zaf.D, re=[z3.real-.1,z1.real+.1], im=[z0.imag-.1,z2.imag+.1], points=5000, axes=ax)
ax.plot([z0.real,z1.real],[z0.imag,z1.imag], 'k')
ax.plot([z1.real,z2.real],[z1.imag,z2.imag], 'k')
ax.plot([z2.real,z3.real],[z2.imag,z3.imag], 'k')
ax.plot([z3.real,z0.real],[z3.imag,z0.imag], 'k')
ax.grid(True)
plt.show()


# %%
print('k=',kmode)
print('------------------------')
print('number of zeros in box', zaf.count_zeros(z0, z1, z2, z3))
zaf.zeros=[]
zeros=zaf.get_zeros(z0, z1, z2, z3, tol = 1.e-6)
print("box: ", z0, z1, z2, z3)
print("Zeros of D in box")
for z in zeros:
    print(z)
    if mp.fabs(zaf.D(z)) > 1e-6:
        print("Value of D at zero ",zaf.D(z))
zero_max=zeros[np.argmax(np.imag(zeros))]
zeros = sorted(zeros, key=lambda z: np.imag(z))


# %%
# Compute least damped zeros of D for given k value with Newton's method
# A good initial guess is essential to get the correct zero
# This initial guess might need to be adjusted for different settings

# Compute negative frequency mode
z_realm = wce/2 - np.sqrt(wce**2/4 + wp**2) #-wce
z_realm = zaf.compute_zeros_newton(z_realm, tol=1e-6, maxiter=100, verbose=0)
print("R mode: ", z_realm)

# Compute high frequency mode
z_realp = wce/2 + np.sqrt(wce**2/4 + wp**2)
z_realp = zaf.compute_zeros_newton(z_realp, tol=1e-6, maxiter=100, verbose=0)
print("L mode: ", z_realp)
# Compute cyclotron modes
z_ic = -1e-6
z_ic = zaf.compute_zeros_newton(z_ic, tol=1e-6, maxiter=100, verbose=0)
z_ec = 0.01
z_ec = zaf.compute_zeros_newton(z_ec, tol=1e-6, maxiter=100, verbose=0)
print("Cyclotron modes: ", -z_ic, z_ec)
# Compute least damped high order mode
z_high_order = wce + (0.2 - 0.2j) * vthe
z_high_order = zaf.compute_zeros_newton(z_high_order, tol=1e-6, maxiter=100, verbose=0)
print("High order mode: ", z_high_order)


# %%
# Find the 4 least damped zeros for different k values

kmax = 4.0 # maximum k value
nk = 40

kk = np.linspace(0.1, kmax, nk)
zeros_array = np.zeros((nk,3), dtype=complex)
box_array = np.zeros((nk,4), dtype=complex)
z_ic_array = np.zeros(nk, dtype=complex)
z_ec_array = np.zeros(nk, dtype=complex)
z_high_order_array = np.zeros(nk, dtype=complex)
z_realm_array = np.zeros(nk, dtype=complex)
z_realp_array = np.zeros(nk, dtype=complex)

# Initial guesses for the zeros
z_realm = wce/2 - np.sqrt(wce**2/4 + wp**2) #-wce
z_realp = wce/2 + np.sqrt(wce**2/4 + wp**2)
z_ic = -1e-6
z_ec = 0.001
z_high_order = wce + (0.005 - 0.1j) * vthe


for i in range(nk):
    kmode = kk[i]
    zaf = zafpy(D, kmode)
    # Compute undamped modes with Newton's method, using the last value as initial guess
    z_realm = z_realm -0.01 * vthe
    z_realm = zaf.compute_zeros_newton(z_realm, tol=1e-10, maxiter=1000)
    z_realm_array[i] = z_realm
    z_realp = zaf.compute_zeros_newton(z_realp, tol=1e-10, maxiter=1000)
    z_realp_array[i] = z_realp
    if simulation[1] == 'two species':
        z_ic = zaf.compute_zeros_newton(z_ic, tol=1e-10, maxiter=1000)
        z_ic_array[i] = -np.conjugate(z_ic)  # store the ion cyclotron mode
    z_ec = zaf.compute_zeros_newton(z_ec, tol=1e-10, maxiter=1000)
    z_ec_array[i] = z_ec
    z_high_order = z_high_order + (0.25 - 0.2j) * vthe
    z_high_order = zaf.compute_zeros_newton(z_high_order, tol=1e-10, maxiter=1000)
    z_high_order_array[i] = z_high_order
    
    print('------------------------')
    print('k=',kmode)
    print("High frequency modes: ", z_realm, z_realp)

    print("Cyclotron modes: ", z_ic_array[i], z_ec)
    print("High order mode: ", z_high_order)
    

# %%
# plot the real and imaginary parts of the zeros for different k
plt.figure()
plt.plot(kk, np.real(z_high_order_array), 'y-', label='Least damped high order mode')
plt.plot(kk, 2*np.ones_like(kk) - np.real(z_high_order_array), 'y-')
plt.plot(kk, np.real(z_ec_array), 'r.-', label='electron cyclotron mode')
if simulation[1] == 'two species': 
    plt.plot(kk, np.real(z_ic_array), 'g.-', label='ion cyclotron mode')
plt.plot(kk, -np.real(z_realm_array), 'b.-', label='R mode')
plt.plot(kk, np.real(z_realp_array), 'c.-', label='L mode')
plt.xlabel('k')
plt.ylabel('Real part of zero')
#plt.legend()
plt.grid()
plt.title('Real part of zeros of D for different k')
plt.show()

plt.figure()
plt.plot(kk, np.imag(z_ec_array), 'r.-', label='electron cyclotron mode')
if simulation[1] == 'two species':  
    plt.plot(kk, np.imag(z_ic_array), 'g.-', label='ion cyclotron mode')
plt.plot(kk, np.imag(z_realm_array), 'b.-', label='R mode')
plt.plot(kk, np.imag(z_realp_array), 'c.-', label='L mode')
plt.plot(kk, np.imag(z_high_order_array), 'y-', label='Least damped high order mode')
plt.xlabel('k')
plt.xlabel('k')
plt.ylabel('Imaginary part of zero')
plt.legend()
plt.grid()
plt.title('Imaginary part of zeros of D for different k')
plt.show()

# Save the zeros to a file
# np.savez('zeros.npz', modes=kk, zeros_array=zeros_array, z_ec_array=z_ec_array, 
#          z_realm_array=z_realm_array, z_realp_array=z_realp_array)


# %%
