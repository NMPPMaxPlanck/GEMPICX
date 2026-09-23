# run this script in parallel (e.g. for 4 cores) with 
# mpirun -n 4 $GEMPICX_DIR/Examples/SupplementaryScripts/PostProcessing/CreateSpaceTimeArrays.py rho 
# assuming that your run directory is in gempic/runs and that you want to do a FFT of rho
# you can replace rho by any field name from the FullDiagnostics. There can be multiple fields
# and they will be saved in separate files

import numpy as np
import argparse
import yt

yt.enable_parallelism()
yt.set_log_level(0) # do not show log output

parser = argparse.ArgumentParser(description='Fields to be read')
parser.add_argument('fields', nargs='+', help='List of fields to process')
args = parser.parse_args()

plotfiles = 'FullDiagnostics/plt_field' + '??????' 

# read times series
ts = yt.load(plotfiles)
nt = len(ts) # number of items in time series
ds = ts[-1]
# check if field is in the field list
for field in args.fields:
    yt_field = ('boxlib', field)
    if yt_field not in ds.field_list:
        raise ValueError(f"Field {field} not found in dataset.")
    
# read in the data for each time slice
storage = {}
for store, ds in ts.piter(storage=storage):
    arr=[]
    ad = ds.all_data()
    data = ds.covering_grid( 0, ds.domain_left_edge, ds.domain_dimensions )
    for field in args.fields:
        field_array = np.array(data['boxlib',field])
        # average over the y and z directions
        if len(np.shape(field_array)) == 3:  # 3D field
            arr.append(np.sum(np.sum(field_array,2),1))
        elif len(np.shape(field_array)) == 2:  # 2D field
            arr.append(np.sum(field_array,1))
        elif len(np.shape(field_array)) == 1:  # 1D field
            arr.append(field_array)
        else:
            raise ValueError(f"Field {field} has unexpected shape: {np.shape(field_array)}")
    
    store.result = arr # np.sum(np.sum(arr,0),0); # we sum over the x and y components
    time = float(ds.current_time)
    
# write space-time array on rank 0 process
if yt.is_root():
    # get space dimensions
    nx, ny, nz = ds.domain_dimensions
    # fill arrays for FFT
    for i, field in enumerate(args.fields):
        arr = np.zeros([nx,nt])
        for data in storage.items():
            arr[:,data[0]] = data[1][i]
        np.save("t_x_array" + field + ".npy",arr)
