# %% [markdown]
# - Convert to jupyter notebook with `jupytext --to ipynb CPDR_xy.py`
# - back to python percent format with `jupytext --to py:percent --opt notebook_metadata_filter=-all CPDR_xy.ipynb`

# %%
# ### Note: This CPDR assumes the xy and z directions are decoupled,
#           which only holds for k∥ (θ=0) or k⊥ (θ=π/2).
#           It will fail for oblique propagation.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import sympy as sp
import yt
yt.set_log_level(0) # do not show log output

# %%
# Input parameters
kmin = 0
kmax = 6
dk = 0.1
theta = 0.5 * np.pi  # degrees, k_vec = [k * sin(theta), 0, k * cos(theta)]
kk = np.arange(kmin, kmax + dk, dk)
mi = 4 # mi/me assume me = 1
me = 1  # me = 1
wce = 1
wpe = 1  # wpe/wce, always use wce=1 
c = 1 # speed of light
# Calculate wci, wpi, and the dispersion relation
wci = 1 / mi
wpi = wpe / np.sqrt(mi)
wp = np.sqrt(wpi**2 + wpe**2)

# Calculate the dispersion relation
w = sp.symbols('w', real=True)
k = sp.symbols('k', real=True)
# actual value of coefficients
S = 1  - wpe**2 / (w**2 - wce**2) - wpi**2 / (w**2 - wci**2)
D =  - wce * wpe**2 / (w*(w**2 - wce**2)) + wci*wpi**2 / (w*(w**2 - wci**2))
P = 1 - wp**2 / w**2 # not used
n2 = (c*k/w)**2

# Transform S, D and n to a polynomial in w by multiplying common denominator
n2pol = sp.simplify(n2*w**2*(w**2 - wce**2)*(w**2 - wci**2))
Spol = sp.simplify(S*w**2*(w**2 - wce**2)*(w**2 - wci**2))
Dpol = sp.simplify(D*w**2*(w**2 - wce**2)*(w**2 - wci**2))
print("Spol = ", Spol)
print("Dpol = ", Dpol)

# Calculate the dispersion relation
DD = sp.Matrix([[Spol - n2pol * sp.cos(theta)**2, -1j*Dpol], [1j*Dpol, Spol - n2pol]])
detD = sp.simplify(DD.det())
print("detD = ", detD)
detD = sp.Poly(detD,w)
print("detD = ", detD)
# Calculate the coefficients of the polynomial
coeffs = detD.all_coeffs()
print("coeffs = ", coeffs)
# Calculate the roots of the polynomial for different k values
roots = []
for k_val in kk:
    # Substitute the value of k into the polynomial
    detD_k = sp.Poly(detD.subs(k, k_val))
    coeffs_k = detD_k.all_coeffs()
    # Calculate the roots of the polynomial
    roots_k = np.roots(coeffs_k)
    # Remove all occurrences of wce in roots by setting them to 0
    roots_k = np.where(np.isclose(roots_k, wce), 0, roots_k)
    # Remove all occurrences of wci from roots by setting them to 0
    roots_k = np.where(np.isclose(roots_k, wci), 0, roots_k)
 
    roots.append(roots_k)
   
# Convert the roots to a numpy array
roots_sorted = np.sort(roots, axis=1)
roots = np.array(roots_sorted)
# Plot the dispersion relation
plt.figure(figsize=(10, 6))
plt.plot(kk, np.real(roots))
# Longitudinal part (carried by Ez) can be calculated analytically for theta = 0 or pi/2
plt.plot(kk, np.sqrt(wp**2 + (c*kk)**2 * np.sin(theta)))
plt.title('Dispersion Relation')
plt.xlabel('k')
plt.ylabel('Frequency (w)')
plt.grid()
plt.xlim(kmin, kmax)
plt.ylim(0, 6)

plt.show()

# Save the roots to a file
# np.savetxt('roots.txt', roots, delimiter=',')
 

# %%
# compute fields

root_number = 8
k_val = kk[root_number]
print("k=", k_val, "w=", np.real(roots[root_number,:]))
w_val = np.real(roots[root_number, 8])
print("k=", k_val, "w=", w_val)
print("Ey/Ex: -1j*", (S.subs(w, w_val)-(c*k_val/w_val)**2 * sp.cos(theta))/D.subs(w, w_val))
print("Bx/Ey: ", -k_val/w_val)
print("By/Ex: ", k_val/w_val)
print("Jx/Ex: ", w_val-k_val**2/w_val)
print("Jy/Ey: ", w_val-k_val**2/w_val)

# %%
# This is for comparing the dispersion relation with the simulation
# A time-space array needs to be created by averaging over the x and y dimensions
# This is done with the CreateSpaceTimeArrays.py script

# Load the array from the file
field = 'Ez' # 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Jx', 'Jy'
arr = np.load("t_x_array" + field + ".npy")
 
ts = yt.load('./FullDiagnostics/plt_field??????')
ntz = len(ts) # number of items in time series

# Load the dataset
ds = ts[-1] # last time step
time = ds.current_time
x_left = np.array(ds.domain_left_edge)
x_right = np.array(ds.domain_right_edge)
L = x_right - x_left


# apply the hann filter
hann = np.hanning(ntz)
arr = arr * hann
# FFT in space and time
arrfft = np.fft.fftn(arr)
# normalize and transpose FFT array for plots
arrfftnorm = np.transpose(np.abs(arrfft))/np.abs(arrfft).max()
[N,L] = arrfftnorm.shape 

T = time # last current time

# define frequency (om) and wave number (kx) values
Lmax = int(L/2)
Nmax = int(N/2)
om = 2*np.pi/T * np.arange(Nmax)
kz = 2*np.pi/Lz * np.arange(Lmax)

lvls = np.logspace(-6, 0, 30)

plt.contourf(kz, om, arrfftnorm [0:Nmax, 0:Lmax], cmap="plasma",norm=colors.LogNorm(), levels=lvls, extend="both")
plt.xlim([0, 6])
plt.ylim([0, 5])
# plot solutions of cold plasma dispersion relation
plt.xlabel('$k_x$')
plt.ylabel('Frequency (w)')    
if field == 'Ex':
    # dispersion relation along kx (= kperp)
    plt.plot(kk, np.real(roots), color = 'black', ls ='dotted')
    plt.title('Dispersion relation from simulation (Ex contour plot)')
elif field == 'Ez':
    # dispersion relation along kz (= kpar)
    plt.plot(kk, np.sqrt(wp**2 + (c*kk)**2 * np.sin(theta)), color = 'black', ls ='dotted')
    plt.title('Dispersion relation from simulation (Ez contour plot)')
else:
    print("Field not supported")
plt.show()

# %%
