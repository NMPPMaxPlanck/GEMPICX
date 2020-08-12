
import matplotlib.pylab as plt
import numpy as np
import yt

###############################################################################
# 1) Read in Norm-Data

filename = "/home/irene/build/gempic/PIC_save_tmp_sampler_2.output" # file with 2 processes
filename_ref = "/home/irene/build/gempic/PIC_save_tmp_sampler_1.output" # file with 1 process

path_2 =  '/home/irene/build/gempic/ParticlePlotfiles/Weibel_sampler_2' # path to folder with data with 2 pr
path_1 =  '/home/irene/build/gempic/ParticlePlotfiles/Weibel_sampler_1' # path to folder with data with 1 pr

# Read in values with 2 processes
n_lines = sum(1 for line in open(filename))

t = np.empty(n_lines-1)
E1 = np.empty(n_lines-1)
E2 = np.empty(n_lines-1)
E3 = np.empty(n_lines-1)
B1 = np.empty(n_lines-1)
B2 = np.empty(n_lines-1)
B3 = np.empty(n_lines-1)
kin = np.empty(n_lines-1)
mom1 = np.empty(n_lines-1)
mom2 = np.empty(n_lines-1)
mom3 = np.empty(n_lines-1)

file = open(filename, 'r')

line_num = 0
for line in file.readlines():
    if line_num > 0 :
        vals = line.rstrip().split(' ')
        t[line_num-1] = vals[0]
        E1[line_num-1] = vals[1]
        E2[line_num-1] = vals[2]
        E3[line_num-1] = vals[3]
        B1[line_num-1] = vals[4]
        B2[line_num-1] = vals[5]
        B3[line_num-1] = vals[6]
        kin[line_num-1] = vals[7]
        mom1[line_num-1] = vals[8]
        mom2[line_num-1] = vals[9]
        mom3[line_num-1] = vals[10]
    line_num = line_num + 1


    
# Read in values with 1 process
n_lines_ref = sum(1 for line in open(filename_ref))

t_ref = np.empty(n_lines_ref-1)
E1_ref = np.empty(n_lines_ref-1)
E2_ref = np.empty(n_lines_ref-1)
E3_ref = np.empty(n_lines_ref-1)
B1_ref = np.empty(n_lines_ref-1)
B2_ref = np.empty(n_lines_ref-1)
B3_ref = np.empty(n_lines_ref-1)
kin_ref = np.empty(n_lines_ref-1)
mom1_ref = np.empty(n_lines_ref-1)
mom2_ref = np.empty(n_lines_ref-1)
mom3_ref = np.empty(n_lines_ref-1)

file = open(filename_ref, 'r')

line_num = 0
for line in file.readlines():
    if line_num > 0 :
        vals = line.rstrip().split(' ')
        t_ref[line_num-1] = vals[0]
        E1_ref[line_num-1] = vals[1]
        E2_ref[line_num-1] = vals[2]
        E3_ref[line_num-1] = vals[3]
        B1_ref[line_num-1] = vals[4]
        B2_ref[line_num-1] = vals[5]
        B3_ref[line_num-1] = vals[6]
        kin_ref[line_num-1] = vals[7]
        mom1_ref[line_num-1] = vals[8]
        mom2_ref[line_num-1] = vals[9]
        mom3_ref[line_num-1] = vals[10]
    line_num = line_num + 1
    
#
    
min_n_lines = np.min([n_lines, n_lines_ref])

###############################################################################
# 2) Plot Fields

field = 'B3' # options: E1 E2 E3 B1 B2 B3 kin mom1 mom2 mom3

plt.figure()
exec("plt.semilogy(t[0:(min_n_lines-1)]," + field + "[0:(min_n_lines-1)],'r-',t_ref[0:(min_n_lines-1)]," + field + "_ref[0:(min_n_lines-1)],'b-', linewidth=1)")
plt.legend((field + ' 2 proc', field + ' 1 proc'))

plt.figure()
exec("plt.semilogy(t[0:(min_n_lines-1)]," + field + "[0:(min_n_lines-1)]-" + field + "_ref[0:(min_n_lines-1)],'b-', linewidth=1)")
plt.legend((field + ' error'))

###############################################################################
# 3) Compute first occurence of error (fields)

E1_index = np.argmax(E1[0:(min_n_lines-1)]-E1_ref[0:(min_n_lines-1)]>0)
E2_index = np.argmax(E2[0:(min_n_lines-1)]-E2_ref[0:(min_n_lines-1)]>0)
E3_index = np.argmax(E3[0:(min_n_lines-1)]-E3_ref[0:(min_n_lines-1)]>0)

B1_index = np.argmax(B1[0:(min_n_lines-1)]-B1_ref[0:(min_n_lines-1)]>0)
B2_index = np.argmax(B2[0:(min_n_lines-1)]-B2_ref[0:(min_n_lines-1)]>0)
B3_index = np.argmax(B3[0:(min_n_lines-1)]-B3_ref[0:(min_n_lines-1)]>0)

kin_index = np.argmax(kin[0:(min_n_lines-1)]-kin_ref[0:(min_n_lines-1)]>0)

mom1_index = np.argmax(mom1[0:(min_n_lines-1)]-mom1_ref[0:(min_n_lines-1)]>0)
mom2_index = np.argmax(mom2[0:(min_n_lines-1)]-mom2_ref[0:(min_n_lines-1)]>0)
mom3_index = np.argmax(mom3[0:(min_n_lines-1)]-mom3_ref[0:(min_n_lines-1)]>0)

print('Field | First occurence of error | error')
print('E1    | ' + str(E1_index) + '                      | ' + str(E1[E1_index]-E1_ref[E1_index]))
print('E2    | ' + str(E2_index) + '                      | ' + str(E2[E2_index]-E2_ref[E2_index]))
print('E3    | ' + str(E3_index) + '                      | ' + str(E3[E3_index]-E3_ref[E3_index]))
print('B1    | ' + str(B1_index) + '                      | ' + str(B1[B1_index]-B1_ref[B1_index]))
print('B2    | ' + str(B2_index) + '                      | ' + str(B2[B2_index]-B2_ref[B2_index]))
print('B3    | ' + str(B3_index) + '                      | ' + str(B3[B3_index]-B3_ref[B3_index]))
print('kin   | ' + str(kin_index) + '                      | ' + str(kin[kin_index]-kin_ref[kin_index]))
print('mom1  | ' + str(mom1_index) + '                      | ' + str(mom1[mom1_index]-mom1_ref[mom1_index]))
print('mom2  | ' + str(mom2_index) + '                      | ' + str(mom2[mom2_index]-mom2_ref[mom2_index]))
print('mom3  | ' + str(mom3_index) + '                      | ' + str(mom3[mom3_index]-mom3_ref[mom3_index]))

###############################################################################
# 4) Plot a given field and error for a given step and given slice
field = 'B_z' # E_x E_y E_z B_x B_y B_z
step_num = B3_index
slice_dir = 'z'
slice_index = 2

# Read data
step = '0'*(5-len(str(step_num))) + str(step_num)

ds_1 = yt.load(path_1+step)
ds_2 = yt.load(path_2+step)

data_1 = ds_1.covering_grid( 0, ds_1.domain_left_edge, ds_1.domain_dimensions )
data_2 = ds_2.covering_grid( 0, ds_2.domain_left_edge, ds_2.domain_dimensions )

field_1 = data_1['boxlib',field]
field_2 = data_2['boxlib',field]

if slice_dir == 'z':
    plt.matshow(field_1[:,:,slice_index])
    plt.colorbar()
    plt.matshow(field_2[:,:,slice_index])
    plt.colorbar()
    plt.matshow(np.abs(field_1[:,:,slice_index]-field_2[:,:,slice_index]))
elif slice_dir == 'y':
    plt.matshow(field_1[:,slice_index,:])
    plt.colorbar()
    plt.matshow(field_2[:,slice_index,:])
    plt.colorbar()
    plt.matshow(np.abs(field_1[:,slice_index,:]-field_2[:,slice_index,:]))
elif slice_dir == 'x':
    plt.matshow(field_1[slice_index,:,:])
    plt.colorbar()
    plt.matshow(field_2[slice_index,:,:])
    plt.colorbar()
    plt.matshow(np.abs(field_1[slice_index,:,:]-field_2[slice_index,:,:]))
plt.colorbar()
###############################################################################
# 5) Check particle data at ocurrence of first error (change step_num to check other timesteps)

step_num = np.min((E1_index, E2_index, E3_index, B1_index, B2_index, B3_index, kin_index, mom1_index, mom2_index, mom3_index))
#step_num = 2
step = '0'*(5-len(str(step_num))) + str(step_num)

# Load data
ds_1 = yt.load(path_1+step)
ds_2 = yt.load(path_2+step)

data_1 = ds_1.covering_grid( 0, ds_1.domain_left_edge, ds_1.domain_dimensions )
data_2 = ds_2.covering_grid( 0, ds_2.domain_left_edge, ds_2.domain_dimensions )

particle_position_x_1 = data_1['electrons','particle_position_x']
particle_position_y_1 = data_1['electrons','particle_position_y']
particle_position_z_1 = data_1['electrons','particle_position_z']

particle_position_x_2 = data_2['electrons','particle_position_x']
particle_position_y_2 = data_2['electrons','particle_position_y']
particle_position_z_2 = data_2['electrons','particle_position_z']
    
particle_vx_1 = data_1['electrons','particle_vx']
particle_vy_1 = data_1['electrons','particle_vy']
particle_vz_1 = data_1['electrons','particle_vz']

particle_vx_2 = data_2['electrons','particle_vx']
particle_vy_2 = data_2['electrons','particle_vy']
particle_vz_2 = data_2['electrons','particle_vz']

print('Max Error of particle positions (x,y,z) at timestep ' + str(step_num))
print(np.max(np.abs(np.sort(np.array(particle_position_x_1))-np.sort(np.array(particle_position_x_2)))))
print(np.max(np.abs(np.sort(np.array(particle_position_y_1))-np.sort(np.array(particle_position_y_2)))))
print(np.max(np.abs(np.sort(np.array(particle_position_z_1))-np.sort(np.array(particle_position_z_2)))))

print('Max Error of particle velocities (vx,vy,vz) at timestep ' + str(step_num))
print(np.max(np.abs(np.sort(np.array(particle_vx_1))-np.sort(np.array(particle_vx_2)))))
print(np.max(np.abs(np.sort(np.array(particle_vy_1))-np.sort(np.array(particle_vy_2)))))
print(np.max(np.abs(np.sort(np.array(particle_vz_1))-np.sort(np.array(particle_vz_2)))))

# Conclusions after testing steps:

# At time step 1: no error
# At time step 2: error in vx,vy,vz
# At time step 3: error in vx,vy,vz and z
# At time step 4: error in vx,vy,vz and y, z
# At time step 5: error in all

###############################################################################
# 6) Development of MAX and MEAN error in particle position and velocity over time
steps_num = [5,55,105,155,205,255,305,355,405,455,505,555,605,655,705,755,805,855,905]

x_max = np.empty(np.size(steps_num))
v_max = np.empty(np.size(steps_num))

x_mean = np.empty(np.size(steps_num))
v_mean = np.empty(np.size(steps_num))

for i in range(np.size(steps_num)):
    step = '0'*(5-len(str(steps_num[i]))) + str(steps_num[i])
    
    ds_1 = yt.load(path_1+step)
    ds_2 = yt.load(path_2+step)

    data_1 = ds_1.covering_grid( 0, ds_1.domain_left_edge, ds_1.domain_dimensions )
    data_2 = ds_2.covering_grid( 0, ds_2.domain_left_edge, ds_2.domain_dimensions )

    particle_position_x_1 = data_1['electrons','particle_position_x']
    particle_position_y_1 = data_1['electrons','particle_position_y']
    particle_position_z_1 = data_1['electrons','particle_position_z']

    particle_position_x_2 = data_2['electrons','particle_position_x']
    particle_position_y_2 = data_2['electrons','particle_position_y']
    particle_position_z_2 = data_2['electrons','particle_position_z']
    
    x1_max = np.max(np.abs(np.sort(np.array(particle_position_x_1))-np.sort(np.array(particle_position_x_2))))
    x2_max = np.max(np.abs(np.sort(np.array(particle_position_y_1))-np.sort(np.array(particle_position_y_2))))
    x3_max = np.max(np.abs(np.sort(np.array(particle_position_z_1))-np.sort(np.array(particle_position_z_2))))
    
    x1_mean = np.mean(np.abs(np.sort(np.array(particle_position_x_1))-np.sort(np.array(particle_position_x_2))))
    x2_mean = np.mean(np.abs(np.sort(np.array(particle_position_y_1))-np.sort(np.array(particle_position_y_2))))
    x3_mean = np.mean(np.abs(np.sort(np.array(particle_position_z_1))-np.sort(np.array(particle_position_z_2))))
    
    particle_vx_1 = data_1['electrons','particle_vx']
    particle_vy_1 = data_1['electrons','particle_vy']
    particle_vz_1 = data_1['electrons','particle_vz']

    particle_vx_2 = data_2['electrons','particle_vx']
    particle_vy_2 = data_2['electrons','particle_vy']
    particle_vz_2 = data_2['electrons','particle_vz']
    
    v1_max = np.max(np.abs(np.sort(np.array(particle_vx_1))-np.sort(np.array(particle_vx_2))))
    v2_max = np.max(np.abs(np.sort(np.array(particle_vy_1))-np.sort(np.array(particle_vy_2))))
    v3_max = np.max(np.abs(np.sort(np.array(particle_vz_1))-np.sort(np.array(particle_vz_2))))
    
    v1_mean = np.max(np.abs(np.sort(np.array(particle_vx_1))-np.sort(np.array(particle_vx_2))))
    v2_mean = np.max(np.abs(np.sort(np.array(particle_vy_1))-np.sort(np.array(particle_vy_2))))
    v3_mean = np.max(np.abs(np.sort(np.array(particle_vz_1))-np.sort(np.array(particle_vz_2))))
    
    x_max[i] = np.max((x1_max,x2_max,x3_max))
    v_max[i] = np.max((v1_max,v2_max,v3_max))
    
    x_mean[i] = np.mean((x1_mean,x2_mean,x3_mean))
    v_mean[i] = np.mean((v1_mean,v2_mean,v3_mean))

plt.figure()
plt.semilogy(t[steps_num],x_max,'-r', t[steps_num],v_max, '-b')
plt.legend(('max x error', 'max v error'))

plt.figure()
plt.semilogy(t[steps_num],x_mean,'-r', t[steps_num],v_mean, '-b')
plt.legend(('mean x error', 'mean v error'))