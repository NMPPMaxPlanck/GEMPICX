# run this script in parallel (e.g. for 4 cores) with 
# mpirun -n 4 ../Examples/SupplementaryScripts/PostProcessing/CreateSpaceTimeArrays.py rho 
# assuming that your run directory is in gempic/runs and that you want to do a FFT or rho
# you can replace rho by any field name from the FullDiagnostics

import numpy as np
import argparse
import yt

yt.enable_parallelism()
yt.set_log_level(0) # do not show log output

#comm = mpi4py.MPI.COMM_WORLD
#rank = comm.Get_rank()

parser = argparse.ArgumentParser(description='Files to be read')
#parser.add_argument('files')
parser.add_argument('field')

args = parser.parse_args()

plotfiles = 'FullDiagnostics/plt_field' + '??????' 

# read times series
ts = yt.load(plotfiles)
ntz = len(ts) # number of items in time series
# ds = ts[-1]
# print(ds.field_list)
    
# read in the data for each time slice
storage = {}
for store, ds in ts.piter(storage=storage):
    ad = ds.all_data()
    data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions )
    arr = np.array(data['boxlib',args.field])
    store.result = np.sum(np.sum(arr,2),1) # we sum over the y and z components
    time = float(ds.current_time)
    
# write space-time array on rank 0 process
if yt.is_root():
    # get space dimensions
    nx, ny, nz = ds.domain_dimensions
    x_left = np.array(ds.domain_left_edge)
    x_right = np.array(ds.domain_right_edge)
    L = x_right - x_left
    # fill array for FFT
    arr = np.zeros([nx,ntz])
    for data in storage.items():
        arr[:,data[0]] = data[1]
    np.save("t_x_array.npy",arr)
