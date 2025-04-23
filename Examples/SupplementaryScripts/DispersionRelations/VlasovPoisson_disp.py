# %% [markdown]
# - Convert to jupyter notebook with `jupytext --to ipynb VlasovPoisson_disp.py`
# - back to python percent format with `jupytext --to py:percent --opt notebook_metadata_filter=-all VlasovPoisson_disp.ipynb`

# %%
from zafpy import *
import numpy as np
import matplotlib.pyplot as plt
import cmath

# %%
# Define dispersion relation
n0 = 1   # density of equilibrium
testcase = 'BumpOnTail'
#testcase = 'TwoStream stable'
#testcase = 'TwoStream unstable'
if testcase == 'TwoStream stable':
    v1 = 1.3  # Average velocity of first gaussian
    v2 = -1.3  # Average velocity of second gaussian#
    vth1 = 1  # Thermal velocity of first gaussian
    vth2 = 1  # Thermal velocity of second gaussian
    w1 = 0.5  # weight of first gaussian
    w2 = 0.5  # weight of second gaussian    
    kmode = 0.2
if testcase == 'TwoStream unstable':
    v1 = 2.4  # Average velocity of first gaussian
    v2 = -2.4  # Average velocity of second gaussian
    vth1 = 1  # Thermal velocity of first gaussian
    vth2 = 1  # Thermal velocity of second gaussian
    w1 = 0.5  # weight of first gaussian
    w2 = 0.5  # weight of second gaussian
    kmode = 0.2
if testcase == 'BumpOnTail':    
    v1 = 0  # Average velocity of first gaussian
    vth1 = 1  # Thermal velocity of first gaussian
    w1 = 0.9  # weight of first gaussian
    v2 = 4.5  # Average velocity of second gaussian
    vth2 = 0.5  # Thermal velocity of second gaussian
    w2 = 0.1  # weight of second gaussian
    kmode = 0.3
omegap = 1 # Plasma frequency
eps0 = 1
e = 1  # charge

# Dispersion relation in symbolic variables for the sum of two gaussians
# Depending on the parameters this can be used for  two stream instability or bump-on-tail instability
def D(omega,k):
    return 1 + (omegap/k)**2*( w1/vth1**2 * (1+ ((omega/k-v1)/(vth1*sp.sqrt(2)))*Z((omega/k-v1)/(vth1*sp.sqrt(2)))) 
             +  w2/vth2**2 * (1 + (omega/k-v2)/(vth2*sp.sqrt(2))*Z((omega/k-v2)/(vth2*sp.sqrt(2)))))

def N(omega,k):
    return (1/2)*(n0*e)/(k*k*eps0*sp.sqrt(2))*(w1/vth1*Z((omega/k-v1)/(vth1*sp.sqrt(2))) + w2/vth2*Z((omega/k-v2)/(vth2*sp.sqrt(2))))


# %% [markdown]
# - In order to find the roots of the dispersion function for given k, which is a complex function we plot it with mpmath.cplot
# - the complex argument (phase) is shown as color (hue) and the magnitude is show as brightness.
#   - This means that colors change quickly around a zero, which can be recognized as a black point with changing colors around 
#   - the white high brightness regions correspond to very large values, where contour integrals are hard to approximate numerically and so should be avoided

# %%
zaf=zafpy(D,kmode,N)
xmin= -3
xmax= 3
ymin = -1
ymax = 1
fig = plt.figure()
ax = plt.subplot(111)
mp.cplot(zaf.D, re=[xmin,xmax], im=[ymin,ymax], points=7000, axes=ax)
ax.grid()
ax.set_xticks(np.linspace(xmin,xmax,10))
ax.set_yticks(np.linspace(ymin,ymax,9))
print("Dispersion function for k = ", kmode )


# %%
# Choose box where zeros are searched for (need to be positively oriented)
z0 = -1.7 - 0.5j
z1 = 1.9 - 0.5j
z2 = 1.9 + 0.5j
z3 = -1.7 + 0.5j

ax.plot([z0.real,z1.real],[z0.imag,z1.imag], 'k')
ax.plot([z1.real,z2.real],[z1.imag,z2.imag], 'k')
ax.plot([z2.real,z3.real],[z2.imag,z3.imag], 'k')
ax.plot([z3.real,z0.real],[z3.imag,z0.imag], 'k')
plt.show()


# %%
print('number of zeros in box', zaf.count_zeros(z0, z1, z2, z3))
zaf.zeros=[]
zeros=zaf.get_zeros(z0, z1, z2, z3)
print("Zeros of D in box")
for z in zeros:
    print(z)
    if mp.fabs(zaf.D(z)) > 1e-12:
        print("Value of D at zero ",zaf.D(z))
zero_max=zeros[np.argmax(np.imag(zeros))]
print('------------------------')
print('k=',kmode)
print('zero with largest imaginary part (omega_j):', zero_max)
# compute polar expression of complex number (need to add pi to phase for our def)
print ('N_over_Dprime: (r, phi) =', np.array(cmath.polar(zaf.N_over_Dprime(zero_max)))-[0,np.pi])

# %%
