# %% [markdown]
# - Convert to jupyter notebook with `jupytext --to ipynb Weibel_disp3D.py`
# - back to python percent format with `jupytext --to py:percent --opt notebook_metadata_filter=-all Weibel_disp3D.ipynb`

# %%
from zafpy import *
import numpy as np
import matplotlib.pyplot as plt
import cmath

# %%
# Define dispersion relation in 3D for up to two Maxwellians with drifts in vx and vy
# test_case = "Weibel"
test_case = "Streaming Weibel"
if test_case == "Weibel":
    alpha = 1 # respective weight of two Maxwellians (0 <= alpha <= 1)
    vth1 = [0.02/np.sqrt(2), 1]  # Thermal velocity  beam (parallel)
    vth2 = [np.sqrt(12)*vth1[0], 1] # Thermal velocity  beam (perpendicular)
    u1 = [0, 0] # streaming velocity in x
    u2 = [0, 0] # Streaming velocity in y
    omegap=[1, 1] # Plasma frequency
    kmode = 1.25 # wave number
elif test_case == "Streaming Weibel": 
    alpha = 1/6 # respective weight of two Maxwellians (0 <= alpha <= 1)
    vth1 = [0.1/np.sqrt(2), 0.1/np.sqrt(2)]  # Thermal velocity  beam (parallel)
    vth2 = [0.1/np.sqrt(2), 0.1/np.sqrt(2)] # Thermal velocity  beam (perpendicular)
    u1 = [0, 0] # streaming velocity in x
    u2 = [0.5, -0.1] # Streaming velocity in y
    omegap=[1, 1] # Plasma frequency  
    kmode = .2 # wave number
c = 1 # speed of light

# Dispersion relation in symbolic variables (here k = kx)
# Note that Dxx gives the Langmuir waves and Dxy = 0 if there is no longitudinal drift (then the zeros of D are the combined zeros of Dxx, Dyy and Dzz)
# For Weibel and streaming Weibel, the zeros we want are those of Dyy
def D(omega,k):
    zeta = [omega / (sp.sqrt(2) * k * vth1[0]), omega / (sp.sqrt(2) * k * vth1[1])]
    Dxx = (omega/c)**2 * (1 + alpha * (omegap[0]/(k*vth1[0]))**2  * ( 1.0 + zeta[0] * Z(zeta[0]))
                          + (1-alpha) * (omegap[1]/(k*vth1[1]))**2  * ( 1.0 + zeta[1] * Z(zeta[1])))
    Dyy = (omega/c)**2 - k**2 - (alpha * (omegap[0]/c)**2 * (1 - ((u2[0]/vth1[0])**2 + (vth2[0]/vth1[0])**2) * ( 1.0 + zeta[0] * Z(zeta[0])))
                            + (1- alpha) * (omegap[1]/c)**2 * (1 - ((u2[1]/vth1[1])**2 + (vth2[1]/vth1[1])**2) * ( 1.0 + zeta[1] * Z(zeta[1]))))
    Dxy = (omega/c)**2 * (alpha * (omegap[0] / (k * vth1[0]))**2 * (k * u1[0] / omega) * ( 1.0 + zeta[0] * Z(zeta[0]))
                          - (1 - alpha) *  (omegap[1] / (k * vth1[1]))**2 * (k * u1[1] / omega) * ( 1.0 + zeta[1] * Z(zeta[1])))
    Dzz = (omega/c)**2 - k**2 - (alpha * (omegap[0]/c)**2 * (1 - (vth2[0]/vth1[0])**2 * ( 1.0 + zeta[0] * Z(zeta[0])))
                                 + (1 - alpha) * (omegap[1]/c)**2 * (1 - (vth2[1]/vth1[1])**2 * ( 1.0 + zeta[1] * Z(zeta[1]))))
    
    return (Dxx * Dyy - Dxy**2) * Dzz


# %% [markdown]
# - In order to find the roots of the dispersion function for given k, which is a complex function we plot it with mpmath.cplot
# - the complex argument (phase) is shown as color (hue) and the magnitude is show as brightness.
# - This means that zeros change quickly around a zero and the white region correspond to very large value, where contours integrals are hard to approximate numerically and so should be avoided

# %%
zaf=zafpy(D,kmode)
xmin= -.1
xmax= .1
ymin = -.05
ymax = 0.075
#mp.cplot(zaf.D, re=[xmin,xmax], im=[ymin,ymax], points=10000)
fig = plt.figure()
ax = plt.subplot(111)
mp.cplot(zaf.D, re=[xmin,xmax], im=[ymin,ymax], points=7000, axes=ax)
ax.grid()
ax.set_xticks(np.linspace(xmin,xmax,10))
ax.set_yticks(np.linspace(ymin,ymax,10))
print("Dispersion function for k = ", kmode )

# %%
# Choose box where zeros are searched for (need to be positively oriented)
z0 = -0.01 - 0.01j
z1 = 0.07 - 0.04j
z2 = 0.07 + 0.02j
z3 = -0.01 + 0.05j

ax.plot([z0.real,z1.real],[z0.imag,z1.imag], 'k')
ax.plot([z1.real,z2.real],[z1.imag,z2.imag], 'k')
ax.plot([z2.real,z3.real],[z2.imag,z3.imag], 'k')
ax.plot([z3.real,z0.real],[z3.imag,z0.imag], 'k')
plt.show()

# %% [markdown]
# ### Numerical evaluation of the zeros using contour integrals
# - we define quadrilateral boxes by 4 complex numbers (z0,z1,z2,z3) ordered anti-clockwise on which contours integrals are computed
# - We need to use small boxes around the zeros to avoid going to high brightness zones where contour integrals will be hard to evaluate numerically

# %%
nzeros = zaf.count_zeros(z0, z1, z2, z3)
print('number of zeros in box', nzeros)
if nzeros > 0.5: 
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
