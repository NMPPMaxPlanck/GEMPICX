# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3.10.6 64-bit
#     language: python
#     name: python3
# ---

# %%
from zafpy import *
import numpy as np
import matplotlib.pyplot as plt
import cmath

# %%
# Define dispersion relation
n0=1   # density of equilibrium
vth=1  # Thermal velocity  beam
omegap=1 # Plasma frequency
eps0 = 1
e = 1  # charge
alpha=1  # Normalization
Te = 10

# Dispersion relation in symbolic variables
def D(omega,k):
    return 1+ alpha*Te*(omegap/vth)**2*(1+ (omega/(k*vth*sp.sqrt(2)))*
    Z(omega/(k*vth*sp.sqrt(2))))


# %% [markdown]
# - In order to find the roots of the dispersion function for given k, which is a complex function we plot it with mpmath.cplot
# - the complex argument (phase) is shown as color (hue) and the magnitude is show as brightness.
# - This means that zeros change quickly around a zero and the white region correspond to very large value, where contours integrals are hard to approximate numerically and so should be avoided

# %%
kmode = 1.
zaf=zafpy(D,kmode)
xmin= -6
xmax= 6
ymin = -4
ymax = 2
#mp.cplot(zaf.D, re=[xmin,xmax], im=[ymin,ymax], points=10000)
fig = plt.figure()
ax = plt.subplot(111)
mp.cplot(zaf.D, re=[xmin,xmax], im=[ymin,ymax], points=7000, axes=ax)
ax.grid()
ax.set_xticks(np.linspace(xmin,xmax,10))
ax.set_yticks(np.linspace(ymin,ymax,9))
print("Dispersion function for k = ", kmode )

# %%
# Choose box where zeros are searched for (need to be positively oriented)
z0 = -6 - 1j
z1 = 6 - 1j
z2 = 6 + 1j
z3 =  -6 + 1j

ax.plot([z0.real,z1.real],[z0.imag,z1.imag], 'k')
ax.plot([z1.real,z2.real],[z1.imag,z2.imag], 'k')
ax.plot([z2.real,z3.real],[z2.imag,z3.imag], 'k')
ax.plot([z3.real,z0.real],[z3.imag,z0.imag], 'k')
display(fig)
plt.clf()

# %% [markdown]
# ### Numerical evaluation of the zeros using contour integrals
# - we define rectangular boxes $[xmin,xmax]\times [ymin,ymax]$) on which contours integrals are computed
# - We need to use small boxes around the zeros to avoid going to high brightness zones where contour integrals will be hard to evaluate numerically

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

# %%
# modulus and phase of complex 0
r = np.abs(3.7288348014877957-0.05833742132118025j)
theta = np.arctan(-0.05833742132118025/3.7288348014877957)
print(r,theta)

# %%
