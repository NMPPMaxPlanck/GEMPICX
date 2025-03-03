import numpy as np
import matplotlib.pyplot as plt
import disprelacalc   # Import the function from disprelacalc.py
from annotate_y_ticks import annotate_y_ticks 
# Input parameters
kmin = 0
kmax = 6
dk = 0.1
theta = 0  # degrees, k_vec = [k * sin(theta), 0, k * cos(theta)]
k = np.arange(kmin, kmax + dk, dk)
mi = 10  # mi/me
wpe = 1  # wpe/wce, always use wce=1

# Calculate wci, wpi, and the dispersion relation
wci = 1 / mi
wpi = wpe / np.sqrt(mi)

mode = "DeFi" 
if mode == "CPDR":
    title = "CPDR Dispersion Relation"
    w =  disprelacalc.CPDR_calc(k, theta, mi, wpe)
    num_roots = 10
    # Plot the results
    plt.figure()
    w10 = w[9, :].squeeze()
    w9 = w[8, :].squeeze()
    w8 = w[7, :].squeeze()
    w7 = w[6, :].squeeze()
    w6 = w[5, :].squeeze()
    plt.plot(k, w10, label='$w_{10}$')
    plt.plot(k, w9, label='$w_{9}$')
    plt.plot(k, w8, label='$w_{8}$')
    plt.plot(k, w7, label='$w_{7}$')
    plt.plot(k, w6, label='$w_{6}$')
elif mode == "DeFi":
    title = "DeFi Dispersion Relation"
    w =  disprelacalc.DeFi_calc(k, theta, mi, wpe)
    num_roots = 8
    # Plot the results
    plt.figure()
    w8 = w[7, :].squeeze()
    w7 = w[6, :].squeeze()
    w6 = w[5, :].squeeze()
    w5 = w[4, :].squeeze()
    plt.plot(k, w8, label='$w_{8}$')
    plt.plot(k, w7, label='$w_{7}$')
    plt.plot(k, w6, label='$w_{6}$')
    plt.plot(k, w5, label='$w_{5}$')   
elif mode == "DK":
    # w = disprelacalc.DK_calc(k, theta, mi, wpe)
    title = "DK Dispersion Relation"
else:
    raise ValueError(f"Unknown mode: {mode}")
# w =  disprelacalc.cpdr_calc(k, theta, mi, wpe)
# num_roots = 10


plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'$kc/\omega_{ce}$', fontsize=14)
plt.ylabel(r'$\omega/\omega_{ce}$', fontsize=14)
plt.title(r'$\theta = ' + str(theta) + r'^\circ$', fontsize=14)

plt.xlim([0, 6])
plt.ylim([0, 6])

# Additional frequency calculations
wce = 1
omegalh = np.sqrt(wce * wci / (1 + wce**2 / wpe**2))
omegauh = np.sqrt(wpe**2 + wce**2)
omega_L = wce / 2 * (np.sqrt(1 + 4 * (wpe / wce)**2) - 1)
omega_R = wce / 2 * (np.sqrt(1 + 4 * (wpe / wce)**2) + 1)

# Example of calling the annotation function
y_values = [omegalh, omegauh, omega_L, omega_R]
y_tick_labels = [r'$\omega_{LH}$', r'$\omega_{UH}$', r'$\omega_{L}$', r'$\omega_{R}$']

# Call the function to annotate y-axis ticks
annotate_y_ticks(y_values, y_tick_labels)

# Save the data
if mode == "CPDR":
    data = np.vstack([k, w10, w9, w8, w7, w6]).T
    if theta == 0:
        np.savetxt('kz_data.csv', data, delimiter=',')
    elif theta == 90:
        np.savetxt('kx_data.csv', data, delimiter=',')
elif mode == "DeFi":
    data = np.vstack([k, w8, w7, w6, w5]).T
    if theta == 0:
        np.savetxt('DeFi_kz_data.csv', data, delimiter=',')
    elif theta == 90:
        np.savetxt('DeFi_kx_data.csv', data, delimiter=',')


