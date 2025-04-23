# %% [markdown]
# - Convert to jupyter notebook with `jupytext --to ipynb LandauIAW_disp.py`
# - back to python percent format with `jupytext --to py:percent --opt notebook_metadata_filter=-all LandauIAW_disp.ipynb`

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
e = -1  # charge
alpha=1  # Normalization
Te = 10

# Dispersion relation in symbolic variables
def D(omega,k):
    return 1+ alpha*Te*(omegap/vth)**2*(1+ (omega/(k*vth*sp.sqrt(2)))*
    Z(omega/(k*vth*sp.sqrt(2))))
def N(omega,k):
    return (1/2)*(n0*e*Te)/(vth*sp.sqrt(2))*Z(omega/(k*vth*sp.sqrt(2)))


# %% [markdown]
# - In order to find the roots of the dispersion function for given k, which is a complex function we plot it with mpmath.cplot
# - the complex argument (phase) is shown as color (hue) and the magnitude is show as brightness.
# - This means that zeros change quickly around a zero and the white region correspond to very large value, where contours integrals are hard to approximate numerically and so should be avoided

# %%
kmode = 1.
zaf=zafpy(D,kmode,N)
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

mp.plot(lambda x: mp.im(zaf.N_over_D_s(x)), [3,4])
mp.plot(lambda x: mp.re(zaf.N_over_D(x)), [-10,10])


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
plt.show()

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
# compute polar expression of complex number (need to add pi to phase for our def)
print ('N_over_Dprime', np.array(cmath.polar(zaf.N_over_Dprime(zero_max)))-[0,np.pi])
#print ('N_over_Dprime', np.array(cmath.polar(zaf.N_over_Dprime(zero_max))))

# %% [markdown]
# ## Exact solution of the linearized problem with inverse Laplace transform
# Computing numerically the inverse Laplace transform is difficult as N/D decays slowly
# However it can be observed that due to the properties of the Z function. 
# The inverse transform is equal to this of the symmetrized function, which decays very fast and hence can be computed numerically at low cost

# %%
# compute numerically integral for inverse Laplace transform
nt = 100
times = np.linspace(0,6,nt)
field = np.zeros(nt,dtype=complex)
fields = np.zeros(nt,dtype=complex)
omega = sp.symbols('omega')
mp.dps = 15
epsilon = 0.04 #* 159

for i in range(nt):
    t = times[i]
    invLap = lambda omega : zaf.N_over_D(omega) * mp.expj(omega*t)
    #field[i] = mp.quad(invLap, np.linspace(-100, 100, 1000)) # does not converge properly
    invLaps = lambda omega : -epsilon * zaf.N_over_D_s(omega) * mp.cos(omega*t) /(2*np.pi)
    fields[i] = 2*mp.quad(invLaps, np.linspace(0, 6, 30)) 
#plt.plot(times,np.imag(field))
plt.plot(times,np.real(fields))
plt.plot(times,np.imag(fields))
plt.legend(['real part','imaginary part','real part anti','imaginary part anti'])


# plot least damped mode obtained previously for comparison
r = 1.87505365
phase = 0.14886984
coef = 2 * epsilon * r
omegar = 3.728834801487808
gamma = -0.0583374213211787
plt.plot(times,(coef*np.cos(omegar*times-phase)*np.exp(gamma*times)))
plt.show()


# %%
