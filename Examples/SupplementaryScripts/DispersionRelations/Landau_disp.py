# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3.10.6 64-bit
#     language: python
#     name: python3
# ---

from zafpy import *
import numpy as np
import matplotlib.pyplot as plt
import cmath

# +
# Define dispersion relation
n0=1   # density of equilibrium
vth=1  # Thermal velocity  beam
omegap=1 # Plasma frequency
eps0 = 1
e = 1  # charge
alpha=1  # Normalization

# Dispersion relation in symbolic variables
def D(omega,k):
    return 1+ alpha*(omegap/vth/k)**2*(1+ (omega/(k*vth*sp.sqrt(2)))*
    Z(omega/(k*vth*sp.sqrt(2))))

def N(omega,k):
    return (1/2)*(n0*e)/(k*k*eps0*vth*sp.sqrt(2))*Z(omega/(k*vth*sp.sqrt(2)))


# -

# - In order to find the roots of the dispersion function for given k, which is a complex function we plot it with mpmath.cplot
# - the complex argument (phase) is shown as color (hue) and the magnitude is show as brightness.
#   - This means that colors change quickly around a zero, which can be recognized as a black point with changing colors around 
#   - the white high brightness regions correspond to very large values, where contour integrals are hard to approximate numerically and so should be avoided

kmode = 0.4
zaf=zafpy(D,kmode,N)
xmin= -3
xmax= 3
ymin = -3
ymax = 1
fig = plt.figure()
ax = plt.subplot(111)
mp.cplot(zaf.D, re=[xmin,xmax], im=[ymin,ymax], points=7000, axes=ax)
ax.grid()
ax.set_xticks(np.linspace(xmin,xmax,10))
ax.set_yticks(np.linspace(ymin,ymax,9))
print("Dispersion function for k = ", kmode )


# +
# Choose box where zeros are searched for (need to be positively oriented)
z0 = 2.8 - 2.7j
z1 = 3.0 - 2.5j
z2 = 1.5 + 0.5j
z3 =  0.2j

ax.plot([z0.real,z1.real],[z0.imag,z1.imag], 'k')
ax.plot([z1.real,z2.real],[z1.imag,z2.imag], 'k')
ax.plot([z2.real,z3.real],[z2.imag,z3.imag], 'k')
ax.plot([z3.real,z0.real],[z3.imag,z0.imag], 'k')
display(fig)

# -

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
print ('N_over_Dprime', np.array(cmath.polar(zaf.N_over_Dprime(zero_max)))-[0,np.pi])

kmode = 0.1
zaf=zafpy(D,kmode,N)
xmin= -1.2
xmax= 1.2
ymin = -0.6
ymax = .5
fig = plt.figure()
ax = plt.subplot(111)
mp.cplot(zaf.D, re=[xmin,xmax], im=[ymin,ymax], points=7000, axes=ax)
ax.grid()
ax.set_xticks(np.linspace(xmin,xmax,10))
ax.set_yticks(np.linspace(ymin,ymax,10))
print("Dispersion function for k = ", kmode )

# +
# Choose box where zeros are searched for (need to be positively oriented)
z0 = 0.6 - 0.6j
z1 = 1.2 - 0.2j
z2 = 1.2 + 0.25j
z3 =  0.14 + 0.12j

ax.plot([z0.real,z1.real],[z0.imag,z1.imag], 'k')
ax.plot([z1.real,z2.real],[z1.imag,z2.imag], 'k')
ax.plot([z2.real,z3.real],[z2.imag,z3.imag], 'k')
ax.plot([z3.real,z0.real],[z3.imag,z0.imag], 'k')
display(fig)
# -

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
print ('N_over_Dprime', np.array(cmath.polar(zaf.N_over_Dprime(zero_max)))-[0,np.pi])


