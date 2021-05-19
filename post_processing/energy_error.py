import matplotlib.pylab as plt
import numpy as np

def compute_total_energy_error(filename):
    # Read in values 

    vals = np.loadtxt(filename)
    t = vals[:,0]
    E1 = vals[:,1]
    E2 = vals[:,2]
    E3 = vals[:,3]
    B1 = vals[:,4]
    B2 = vals[:,5]
    B3 = vals[:,6]
    kin = vals[:,7]
    
    # Compute total energy error
    err = (E1+E2+E3+B1+B2+B3)*127.001709283 + 2*kin
    return err

def return_value(filename, value):
    # Read in values 
    vals = np.loadtxt(filename)
    val = vals[:,value]
    
    return val

path = '/home/irene/Documents/gempic_amrex/data/energy_conservation/higher_precision/'

order_2_dt_8 = compute_total_energy_error(path+'strang_2_dt_8.output')
order_2_dt_4 = compute_total_energy_error(path+'strang_2_dt_4.output')
order_2_dt_2 = compute_total_energy_error(path+'strang_2_dt_2.output')
order_2_dt_1 = compute_total_energy_error(path+'strang_2_dt_1.output')

#order_4_dt_8 = compute_total_energy_error(path+'strang_4_dt_8.output')
#order_4_dt_4 = compute_total_energy_error(path+'strang_4_dt_4.output')
#order_4_dt_2 = compute_total_energy_error(path+'strang_4_dt_2.output')
#order_4_dt_1 = compute_total_energy_error(path+'strang_4_dt_1.output')

plt.figure()
plt.semilogy(return_value(path+'strang_2_dt_8.output',0), np.abs(order_2_dt_8-order_2_dt_8[0]))
plt.semilogy(return_value(path+'strang_2_dt_4.output',0), np.abs(order_2_dt_4-order_2_dt_4[0]))
plt.semilogy(return_value(path+'strang_2_dt_2.output',0), np.abs(order_2_dt_2-order_2_dt_2[0]))
plt.semilogy(return_value(path+'strang_2_dt_1.output',0), np.abs(order_2_dt_1-order_2_dt_1[0]))
plt.legend(['order_2_dt_8','order_2_dt_4','order_2_dt_2','order_2_dt_1'])

plt.semilogy(return_value(path+'strang_4_dt_8.output',0), np.abs(order_4_dt_8-order_4_dt_8[0]))
plt.semilogy(return_value(path+'strang_4_dt_4.output',0), np.abs(order_4_dt_4-order_4_dt_4[0]))
plt.semilogy(return_value(path+'strang_4_dt_2.output',0), np.abs(order_4_dt_2-order_4_dt_2[0]))
plt.semilogy(return_value(path+'strang_4_dt_1.output',0), np.abs(order_4_dt_1-order_4_dt_1[0]))

plt.legend(['order_2_dt_8','order_2_dt_4','order_2_dt_2','order_2_dt_1',
            'order_4_dt_8','order_4_dt_4','order_4_dt_2','order_4_dt_1'])
plt.xlim([0,1])

# B3
#plt.figure()
#plt.semilogy(return_value(path+'strang_2_dt_8.output', 0), return_value(path+'strang_2_dt_8.output', 6))
#plt.semilogy(return_value(path+'strang_2_dt_4.output', 0), return_value(path+'strang_2_dt_4.output', 6))
#plt.semilogy(return_value(path+'strang_2_dt_2.output', 0), return_value(path+'strang_2_dt_2.output', 6))
#plt.semilogy(return_value(path+'strang_2_dt_1.output', 0), return_value(path+'strang_2_dt_1.output', 6))

#plt.semilogy(return_value(path+'strang_4_dt_8.output', 0), return_value(path+'strang_4_dt_8.output', 6))
#plt.semilogy(return_value(path+'strang_4_dt_4.output', 0), return_value(path+'strang_4_dt_4.output', 6))
#plt.semilogy(return_value(path+'strang_4_dt_2.output', 0), return_value(path+'strang_4_dt_2.output', 6))
#plt.semilogy(return_value(path+'strang_4_dt_1.output', 0), return_value(path+'strang_4_dt_1.output', 6))
