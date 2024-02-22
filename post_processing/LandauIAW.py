# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3.11.1 64-bit
#     language: python
#     name: python3
# ---

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yt


# %%
# plot electric energy
tabE=pd.read_csv("reducedDiagnostics/FieldElec.txt",delim_whitespace=True)
#tab.plot(1,2)
t = tabE.values[:,1]
ex2 = tabE.values[:,2]
ey2 = tabE.values[:,3]
ez2 = tabE.values[:,4]
etot = ex2+ey2+ez2
itmax=500
plt.plot(t[:itmax],np.log(ex2[:itmax]))
plt.ylim(-2,4)
Te = 10
coef=2*0.04*3.73*Te # 2.984
#coef=2.8
plt.plot(t[:itmax], np.log((coef*np.cos(3.7288*t[:itmax]-.22)*np.exp(-0.058337*t[:itmax]))**2))
plt.plot(t[:itmax], np.log((coef*np.exp(-0.058337*t[:itmax]))**2))

# plot particle kinetic energy
tabK=pd.read_csv("reducedDiagnostics/PartDiag.txt",delim_whitespace=True)
#tab.plot(1,2)
t = tabK.values[:,1]
kin = tabK.values[:,5]
#plt.plot(t,kin)
print(tabE)
plt.savefig("landauIAW004ppc1000F3Te10")

# %%
