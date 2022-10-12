import matplotlib.pylab as plt
import numpy as np
import yt

ds_init = yt.load("/home/irene/build/gempic/src/field_solvers/testing/rho_init")
ds_end = yt.load("/home/irene/build/gempic/src/field_solvers/testing/rho_end")
ds_Drho = yt.load("/home/irene/build/gempic/src/field_solvers/testing/Drho_x")
ds_HDrho = yt.load("/home/irene/build/gempic/src/field_solvers/testing/HDrho_x")
ds_DtHDrho = yt.load("/home/irene/build/gempic/src/field_solvers/testing/DtHDrho_x")

data_init = ds_init.covering_grid( 0, ds_init.domain_left_edge, ds_init.domain_dimensions )
data_end = ds_end.covering_grid( 0, ds_end.domain_left_edge, ds_end.domain_dimensions )
data_Drho = ds_Drho.covering_grid( 0, ds_Drho.domain_left_edge, ds_Drho.domain_dimensions )
data_HDrho = ds_HDrho.covering_grid( 0, ds_HDrho.domain_left_edge, ds_HDrho.domain_dimensions )
data_DtHDrho = ds_DtHDrho.covering_grid( 0, ds_DtHDrho.domain_left_edge, ds_DtHDrho.domain_dimensions )


rho_init = np.array(data_init['boxlib','rho'])
rho_end = np.array(data_end['boxlib','rho'])
Drho = np.array(data_Drho['boxlib','Drho_x'])
HDrho = np.array(data_HDrho['boxlib','HDrho_x'])
DtHDrho = np.array(data_DtHDrho['boxlib','DtHDrho_x'])

plt.plot(rho_init[:,slice,slice])
plt.plot(rho_end[:,slice,slice])


slice = 16

plt.figure()
plt.plot(rho_init[:,slice,slice])
plt.plot(Drho[:,slice,slice])
plt.figure()
plt.plot(Drho[:,slice,slice])
plt.plot(HDrho[:,slice,slice])
plt.figure()
plt.plot(HDrho[:,slice,slice])
plt.plot(DtHDrho[:,slice,slice])
plt.figure()
plt.plot(DtHDrho[:,slice,slice])
plt.plot(rho_end[:,slice,slice])

vec = HDrho[:,slice,slice]
manual_stencil = (vec - np.roll(vec,-1))/(4*np.pi/32)
plt.figure()
plt.plot(DtHDrho[:,slice,slice])
plt.plot(manual_stencil)