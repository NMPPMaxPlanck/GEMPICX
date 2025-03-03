# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3.11.1 64-bit
#     language: python
#     name: python3
# ---

import numpy as np

# ### Physical constants
# SI units except T in eV

c = 2.9979e8
eps0 = 8.8542e-12
mu0 = 1/(eps0*c**2)
me = 9.1094e-31
mi =  1.6726e-27  # proton mass
e = 1.6e-19 # electron charge

# ### Fundamental parameters
# - density ($n$)
# - Temperature ($T_e$,$T_i$) ions and ions
# - magnetic field ($B_0$)

#plasma = "ionosphere"
plasma = "fusion"
if plasma == "fusion":
    n = 1e20
    Te = 1e4
    Ti = 1e4
    B0 = 2
if plasma == "ionosphere":    
    n = 1e12
    Te = 0.1
    Ti = 0.1
    B0 = 0.35e-4

# ### Derived parameters
# - Thermal velocity $v_{th}= \sqrt{\frac{k_B T}{m}}$ ($k_B=e$ and $T$ in eV)
# - Plasma frequency $\omega_p = \frac{q^2 n}{\epsilon_0 m}$
# - Cyclotron frequency $\omega_c = \frac{q B}{m}$
# - thermal gyroradius $\rho_L = \frac{v_{th}}{\omega_c}$
# - inertial length $d=\frac{c}{\omega_p}$
# - Debye length $\lambda_D = \frac{v_{th}}{\omega_p}$ 
# - Alfven velocity $v_A = \frac{B_0}{\mu_0 m_i n}$
# - plasma $\beta = \frac{2\mu_0 n k_B T}{B_0^2}$

vthe = np.sqrt(e*Te/me)
omegape = np.sqrt(e**2*n/(me*eps0))
omegace = e*B0/me
rLe = vthe/omegace
de = c/omegape
betae = 2*mu0*n*e*Te/B0**2
lambdaD = vthe/omegape
vthi = np.sqrt(e*Ti/mi)
omegapi = np.sqrt(e**2*n/(mi*eps0))
omegaci = e*B0/mi
rLi = vthi/omegaci
di = c/omegapi
betai = 2*mu0*n*e*Ti/B0**2
vA = B0/np.sqrt(mu0*mi*n)

print(f"{lambdaD = :.3e}  (Debye length)")
print(f"{vthe    = :.3e}  (electron thermal velocity)")
print(f"{omegape = :.3e}  (electron plasma frequency)")
print(f"{omegace = :.3e}  (electron cyclotron frequency)")
print(f"{rLe     = :.3e}  (electron thermal Larmor radius)")
print(f"{de      = :.3e}  (electron inertial length / skin depth)")
print(f"{betae   = :.3e}  (ration electron thermal / magnetic energy)")
print(f"{vthi    = :.3e}  (ion thermal velocity)")
print(f"{omegapi = :.3e}  (ion plasma frequency)")
print(f"{omegaci = :.3e}  (ion cyclotron frequency)")
print(f"{rLi     = :.3e}  (ion thermal Larmor radius)")
print(f"{di      = :.3e}  (ion inertial length / skin depth)")
print(f"{betai   = :.3e}  (ration ion thermal / magnetic energy)")
print(f"{vA      = :.3e}  (Alfven velocity)")


print(omegapi/omegaci,omegape/omegace, omegapi/omegaci*omegace/omegape)

np.sqrt(mi/me)


