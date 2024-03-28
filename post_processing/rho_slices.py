# Import needed modules (needs yt>=4 version, do pip install 'yt>=4.0' if needed)
# Classical yt plot examples can be found in Cookbook https://yt-project.org/doc/cookbook
import os
import numpy as np
import matplotlib.pyplot as plt
import yt

# Set working directory
pathname = '.'
pathname_out = pathname + '/processed'
try:
    os.mkdir(pathname_out)
except(FileExistsError):
    pass
os.chdir(pathname)
sim_name = "rho"

# Read data
slices =os.listdir("./FullDiagnostics")
slices.sort()

n_t_step = 0 # the time step; 0 for initial  
plotfile = pathname + '/FullDiagnostics/plt_field{:06d}'.format(n_t_step)
# print the AMReX plotfiles you are going to load
print(plotfile)
ds = yt.load(plotfile)
# print field list
print(ds.field_list)
field_data = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
rho = field_data['boxlib', 'rho']
print(np.shape(rho))
plt.plot((rho)[:,0,0])
plt.xlabel('nx')
plt.ylabel('rho')
plt.show()
# Save the plot as a PDF file
savepath=pathname_out+'/sample_plot.pdf'
plt.savefig(savepath, format='pdf')

