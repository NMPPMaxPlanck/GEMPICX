# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import yt
from yt.frontends.boxlib.data_structures import AMReXDataset
import matplotlib.animation as animation

# %%
# read times series of fields
ts = yt.load('../../build/cpu-debug-3D/Testing/FullDiagnostics/plt_field??????')
# Print number of items in time series
ntz = ts.__len__()
print('number of time slices ', ntz)
# print field list of first time steps. They are the same for all time steps
print("yt field list ",ts[0].field_list)

#%%
fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True)

ds = ts[0]
x_left = np.array(ds.domain_left_edge)
x_right = np.array(ds.domain_right_edge)
field_data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions )
Ex = field_data['boxlib','Ex']
By = field_data['boxlib','By']

x = np.linspace(x_left[2],x_right[2],Ex.shape[2])
line1, = ax[0].plot(x, Ex[0,0,:], label="Ex")
line2, = ax[1].plot(x, By[0,0,:], label="By", color="tab:orange")
#ax.legend(loc="upper right")

ax[0].set_xlim([x_left[2], x_right[2]])
ax[0].set_ylim([-1.1, 1.1])
ax[1].set_ylim([-1.1, 1.1])
ax[1].set_xlabel("$x$")
ax[0].set_ylabel("$E_x$")
ax[1].set_ylabel("$B_y$")

time = float(ds.current_time)
time_text = ax[0].text(0, 1.25, f"t = {time:.3f}", fontsize=15)

def update(i):

    ds = ts[i]
    field_data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions )
    Ex = field_data['boxlib', 'Ex']
    By = field_data['boxlib', 'By']
 
    line1.set_ydata(Ex[0,0,:])
    line2.set_ydata(By[0,0,:])
    
    time = float(ds.current_time)
    time_text.set_text(f"t = {time:.3f}")

    return (line1, line2, time_text)

step = 1
anim = animation.FuncAnimation(fig=fig, func=update, frames=range(0,ntz,step), interval=1)

writervideo = animation.FFMpegWriter(fps=5) 
anim.save('animation_maxwell.mp4', writer=writervideo) 

plt.close()