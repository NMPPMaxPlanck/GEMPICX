# %% [markdown]
# - Convert to jupyter notebook with `jupytext --to ipynb verify_dispersion_relation.py`
# - back to python percent format with `jupytext --to py:percent --opt notebook_metadata_filter=-all verify_dispersion_relation.ipynb`
#

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import sympy as sp  
import yt
yt.set_log_level(0) # do not show log output
GEMPICX_DIR = '../..' # user provided
import sys
sys.path.append(GEMPICX_DIR + '/Examples/SupplementaryScripts/DispersionRelations')
from zafpy import *

# %%
# type of simulation 
# First argument: 'fully kinetic' or 'quasi-neutral'
# Second argument: 'parallel' or 'perpendicular' 
#   (as our simulations are in practice 1D in the x direction, k is kx 
#   and the simulation is parallel if Bx = 1 and parallel if Bz = 1)
# Third argument: 'one species' or 'two species'
simulation = ['fully kinetic', 'parallel', 'two species'] 

# Set working directory where the simulation is run
os.chdir(GEMPICX_DIR + '/runs/DispersionTwoSpeciesParVth0.5')

# %%
# Set physical parameters
if simulation[1] == 'parallel':
        theta = 0 
elif simulation[1] == 'perpendicular':
    theta = 0.5 * np.pi
else:
    raise ValueError("Invalid simulation type. Choose 'parallel' or 'perpendicular'.")
# Define the physical parameters for the dispersion relation which will be used in the simulation
if simulation[2] == 'one species':
    mi = 1.0e12 # mass is set very large
else:
    mi = 4  # actual value for two species
me = 1
B0 = 1  # magnetic field strength in x or z direction
c = 1 # speed of light
vthe = 0.2
vthi = vthe * np.sqrt(me / mi)
# Calculate the plasma frequencies
wce = B0 / me  # assuming e = 1
wpe = 1 / np.sqrt(me) # assuming a unit density 
wci = B0 / mi
wpi = wpe / np.sqrt(mi)
wp = np.sqrt(wpi**2 + wpe**2)
print("Simulation type: ", simulation)
print("wpe = ", wpe, "wpi = ", wpi)
print("wce = ", wce, "wci = ", wci)
print("vthe = ", vthe, "vthi = ", vthi)

#%%
# Cold plasma dispersion relation
# Input parameters for cold plasma dispersion relation
def compute_CPDR(simulation, field, kk):

    if ((field == 'Ex' and simulation[1] == 'parallel') or 
        (field == 'Ez' and simulation[1] == 'perpendicular')):
        # Longitudinal modes can be calculated analytically for theta = 0 or pi/2
        roots = np.array(np.sqrt(wp**2 + (c*kk)**2 * np.cos(theta)))
    else:     
        # define the transverse dispersion relation
        w = sp.symbols('w', real=True)
        k = sp.symbols('k', real=True)
        S=0
        D=0
        if simulation[0] == 'fully kinetic':
            S = 1  - wpe**2 / (w**2 - wce**2) - wpi**2 / (w**2 - wci**2)
            D =  - wce * wpe**2 / (w*(w**2 - wce**2)) + wci*wpi**2 / (w*(w**2 - wci**2))
        elif simulation[0] == 'quasi-neutral':
            S = - wpe**2 / (w**2 - wce**2) - wpi**2 / (w**2 - wci**2)
            D =  - wce * wpe**2 / (w*(w**2 - wce**2)) + wci*wpi**2 / (w*(w**2 - wci**2))
        n2 = (c*k/w)**2

        # Transform S, D and n2 to a polynomegaial in w by multiplying comegamon denominator
        n2pol = sp.simplify(n2*w**2*(w**2 - wce**2)*(w**2 - wci**2))
        Spol = sp.simplify(S*w**2*(w**2 - wce**2)*(w**2 - wci**2))
        Dpol = sp.simplify(D*w**2*(w**2 - wce**2)*(w**2 - wci**2))

        # Calculate the dispersion relation
        DD = sp.Matrix([[Spol - n2pol * np.cos(theta)**2, -1j*Dpol], [1j*Dpol, Spol - n2pol]])
        detD = sp.simplify(DD.det())
        detD = sp.Poly(detD,w)
        # Calculate the coefficients of the polynomial
        coeffs = detD.all_coeffs()
        # Calculate the roots of the polynomial for different k values
        roots = []
        for k_val in kk:
            # Substitute the value of k into the polynomial
            detD_k = sp.Poly(detD.subs(k, k_val))
            coeffs_k = detD_k.all_coeffs()
            # Calculate the roots of the polynomial
            roots_k = np.roots(coeffs_k)
            # Remove all occurrences of wce in roots by setting them to 0 (might also remove some other roots)
            roots_k = np.where(np.isclose(roots_k, wce), 0, roots_k)
            # Remove all occurrences of wci from roots by setting them to 0
            roots_k = np.where(np.isclose(roots_k, wci), 0, roots_k)
            # Append the roots to the list
            roots.append(roots_k)

        # Convert the roots to a numpy array
        roots_sorted = np.sort(roots, axis=1)
        roots = np.array(roots_sorted)

    return roots

plot_CPDR = False
# Plot the cold plasma dispersion relation for debugging purposes
kk = np.linspace(0, 6, 100)  # wave numbers
# Compute the cold plasma dispersion relation
roots = compute_CPDR(simulation, 'Ez', kk)
if plot_CPDR:
    # Plot the dispersion relation
    plt.figure(figsize=(10, 6))
    plt.plot(kk, np.real(roots))
    # Solution along B can be calculated analytically for theta = 0 or pi/2
    if simulation[0] == 'fully kinetic':
        # Cold Plasma dispersion relation along kx (= kpar)
        plt.title('Cold Plasma Dispersion Relation (Parallel)')
        plt.plot(kk, roots, color = 'black', ls ='dotted')
        plt.title('Cold Plasma Dispersion Relation')
    plt.xlabel(r'$kc/\omega_{pe}$')
    plt.ylabel(r'Frequency ($\omega/\omega_{ce}$)')
    plt.grid()
    plt.xlim(kk[0], kk[-1])
    plt.ylim(0, 3)

plt.show()


# %%
# Compute roots of the hot plasma dispersion relation (HPDR) parallel to B
def compute_HPDR(simulation,field,kk):
    """
        Only works for waves parallel to B.

        Computes the roots of the transverse mode dispersion relation for a given plasma simulation type and array of wavenumbers.
        This function solves the dispersion relation for transverse modes in a plasma, using either a one-species or two-species model.
        The roots are found using a root-finding algorithm over a range of frequencies determined by the cyclotron frequencies of the plasma species.
        simulation : list or tuple
            Simulation configuration, where simulation[1] must be either 'one species' or 'two species'.
        zeros_array : ndarray
            A 2D array of shape (len(kk), 5), where each row contains the computed roots (frequencies) for the corresponding wavenumber.
            The columns correspond to different harmonics or branches of the transverse mode. If a root cannot be found, the entry is set to zero.
        - The function relies on several global variables (e.g., wce, wci, wpe, wpi, vthe, vthi, c) that must be defined in the calling scope.
        - Uses Newton's method for root-finding via the zafpy library.
        - For two-species plasmas, both electron and ion contributions are included in the dispersion relation.
        - The number and meaning of roots per wavenumber depend on the plasma species configuration.
        - Initial guesses for the roots are chosen based on physical considerations of the plasma parameters.
    """
    nk = len(kk)

    if field == 'Ex':
        # Dispersion relation of longitudinal modes
        if simulation[2] == 'two species':
            if simulation[0] == 'fully kinetic':
                def D(omega,k):
                    return 1 + (wpe/(k*vthe))**2*(1+ (omega/(k*vthe*sp.sqrt(2))) *
                    Z(omega/(k*vthe*sp.sqrt(2)))) + (wpi/(k*vthi))**2*(1+ (omega/(k*vthi*sp.sqrt(2))) *
                    Z(omega/(k*vthi*sp.sqrt(2)))) 
            elif simulation[0] == 'quasi-neutral':
                def D(omega,k):
                    return (wpe/(k*vthe))**2*(1+ (omega/(k*vthe*sp.sqrt(2))) *
                        Z(omega/(k*vthe*sp.sqrt(2)))) + (wpi/(k*vthi))**2*(1+ (omega/(k*vthi*sp.sqrt(2))) *
                        Z(omega/(k*vthi*sp.sqrt(2))))
        elif simulation[2] == 'one species':
            if simulation[0] == 'fully kinetic':
                def D(omega,k):
                    return 1 + (wpe/(k*vthe))**2*(1+ (omega/(k*vthe*sp.sqrt(2))) *
                    Z(omega/(k*vthe*sp.sqrt(2))))
            elif simulation[0] == 'quasi-neutral':
                def D(omega,k):
                    return (wpe/(k*vthe))**2*(1+ (omega/(k*vthe*sp.sqrt(2))) *
                        Z(omega/(k*vthe*sp.sqrt(2))))
        else:
            raise ValueError("Invalid simulation type. Choose 'one species' or 'two species'.")
    
        
        zeros_array = np.zeros((nk,2), dtype=complex)

        # Initial guesses for the zeros
        z_least = wp
        z_high_order = 0

        for i in range(nk):
            kmode = kk[i]
            zaf = zafpy(D, kmode)
            # Compute undamped modes with Newton's method, using the last value as initial guess
            z_least = zaf.compute_zeros_newton(z_least, tol=1e-6, maxiter=1000)
            zeros_array[i,0] = z_least
            z_high_order = z_high_order + (0.25 - 0.2j) * vthe
            z_high_order = zaf.compute_zeros_newton(z_high_order, tol=1e-10, maxiter=1000)
            zeros_array[i,1] = z_high_order
            
    elif field == 'Ez':
        # Dispersion relation of transverse modes
        # Dispersion relation from Fitzpatrick 
        # (https://farside.ph.utexas.edu/teaching/plasma/Plasmahtml/node95.html) eq 7.98. 
        # Note that in his convention wce = -B0/me and wci = B0/mi, whereas for us both frequencies are positive.
        if simulation[2] == 'two species':
            def D(omega,k):
                return (1 - (k*c)**2/omega**2 + wpe**2/(sp.sqrt(2)*omega*vthe*k)* Z((omega-wce)/(k*vthe*sp.sqrt(2)))
                + wpi**2/(sp.sqrt(2)*omega*vthi*k)* Z((omega+wci)/(k*vthi*sp.sqrt(2))))   
        elif simulation[2] == 'one species':
            def D(omega,k):
                return  1 - (k*c)**2/omega**2 + wpe**2/(sp.sqrt(2)*omega*vthe*k) * Z((omega-wce)/(k*vthe*sp.sqrt(2)))
        else:
            raise ValueError("Invalid simulation type. Choose 'one species' or 'two species'.")
        # Create an array to store the zeros
        zeros_array = np.zeros((nk,8), dtype=complex)

        # Initial guesses for the zeros
        z_realm = wce/2 - np.sqrt(wce**2/4 + wp**2) #-wce
        z_realp = wce/2 + np.sqrt(wce**2/4 + wp**2)
        z_ic = -1e-6 # ion cyclotron mode (electron cyclotron mode if one species)
        z_ec = 1e-3
        z_high_order_i = -wci - 0.01
        z_high_order_e = wce + (0.005 - 0.1j) * vthe

        for i in range(nk):
            kmode = kk[i]
            zaf = zafpy(D, kmode)
            # Compute undamped modes with Newton's method, using the last value as initial guess
            z_realm = z_realm - 0.01 * vthe
            z_realm = zaf.compute_zeros_newton(z_realm, tol=1e-10, maxiter=1000)
            zeros_array[i,0] = -z_realm # negative sign for the left-hand side mode
            z_realp = zaf.compute_zeros_newton(z_realp, tol=1e-10, maxiter=1000)
            zeros_array[i,1] = z_realp
            z_ic = zaf.compute_zeros_newton(z_ic, tol=1e-10, maxiter=1000)
            zeros_array[i,2] = -z_ic
            z_ec = zaf.compute_zeros_newton(z_ec, tol=1e-10, maxiter=1000)
            zeros_array[i,3] = z_ec
            z_high_order_e = z_high_order_e + (0.25 - 0.2j) * vthe
            z_high_order_e = zaf.compute_zeros_newton(z_high_order_e, tol=1e-10, maxiter=1000)
            zeros_array[i,4] = z_high_order_e
            z_high_order_i = z_high_order_i + (0.25 - 0.2j) * vthi
            #z_high_order_i = zaf.compute_zeros_newton(z_high_order_i, tol=1e-10, maxiter=1000)
            zeros_array[i,5] = -z_high_order_i
            # symmetric high order modes
            zeros_array[i,6] = 2 * wce - z_high_order_e
            zeros_array[i,7] = 2 * wci + z_high_order_i

    return zeros_array

print_roots = False
# Print the roots of the hot plasma dispersion relation for debugging purposes
if print_roots:
    simulation = ['fully kinetic', 'parallel', 'two species']
    field = 'Ez'  
    kk = np.linspace(0.1, 4, 40)  # wave numbers
    zeros_array = compute_HPDR(simulation,field,kk)
    nk = len(kk)
    for i in range(nk):
        kmode = kk[i]
        print('------------------------')
        print('k=',kmode)
        print("High frequency modes: ", zeros_array[i,0], zeros_array[i,1])
        print("cyclotron modes: ", zeros_array[i,2], zeros_array[i,3])
        print("High order modes: ", zeros_array[i,4], zeros_array[i,5])

plot_HPDR = True
# Plot the hot plasma dispersion relation for debugging purposes
if plot_HPDR:
    kk = np.linspace(0.1, 4, 40)  # wave numbers
    simulation_HPDR = ['fully kinetic', 'parallel', 'two species']
    # Compute the hot plasma dispersion relation along kx (= kpar)
    # 1) Longitudinal mode
    field = 'Ex'
    roots = compute_HPDR(simulation_HPDR,field,kk)
    # Plot the dispersion relation
    plt.figure(figsize=(10, 6))
    plt.plot(kk, np.real(roots))
    plt.title('Hot Plasma Dispersion Relation Ex')
    plt.xlabel(r'$kc/\omega_{pe}$')
    plt.ylabel(r'Frequency ($\omega/\omega_{ce}$)')
    plt.grid()
    plt.xlim(kk[0], kk[-1])
    plt.ylim(0, 3)
    
    # 2) Transverse mode
    field = 'Ez'
    roots = compute_HPDR(simulation_HPDR,field,kk)
    # Plot the dispersion relation
    plt.figure(figsize=(10, 6))
    plt.plot(kk, np.real(roots))
    plt.title('Hot Plasma Dispersion Relation Ez')
    plt.xlabel(r'$kc/\omega_{pe}$')
    plt.ylabel(r'Frequency ($\omega/\omega_{ce}$)')
    plt.grid()
    plt.xlim(kk[0], kk[-1])
    plt.ylim(0, 3)

plt.show()

    

# %%
# Calculate the Bernstein wave dispersion relation (perpendicular to B)
# This is a simplified version of the Bernstein wave dispersion relation
from scipy.special import iv  # modified bessel function
from scipy.optimize import brentq
# Bernstein wave dispersion function from Bernstein 1958 (Z is approximated)
# Formulas from Fitzpatrick (Plasma phasics: An introduction) available online under the link
# https://farside.ph.utexas.edu/teaching/plasma/lectures1/Plasma2html.html
# chapter waves in warm plasmas
def compute_roots_bernstein(field,kk):
    """
    Computes the roots of the Bernstein wave dispersion relation for a given field ('Ez' or 'Ex') and array of wavenumbers.
    The function solves the dispersion relation for Bernstein waves in a plasma, using either the Ez or Ex field formulation
    as described in Fitzpatrick's plasma physics book. It supports both one-species and two-species plasma models, as specified
    by the global 'simulation' variable. The roots are found using a root-finding algorithm over a range of frequencies determined
    by the cyclotron frequencies of the plasma species.
    Parameters
    ----------
    field : str
        The field component to use in the dispersion relation. Must be either 'Ez' or 'Ex'.
    kk : array_like
        Array of wavenumbers at which to compute the Bernstein wave roots.
    Returns
    -------
    roots_bernstein : ndarray
        A 2D array of shape (n_omega, len(kk)), where n_omega is the number of Bernstein wave branches (harmonics),
        and len(kk) is the number of wavenumbers. Each entry contains the computed root (frequency) for the corresponding
        harmonic and wavenumber. If a root cannot be found, the entry is set to zero.
    Notes
    -----
    - The function relies on several global variables (e.g., wce, wci, wpe, wpi, vthe, vthi, c, simulation) that must be defined
        in the calling scope.
    - Uses the modified Bessel function of the first kind (iv) and root-finding routines from scipy.optimize.
    - The number of harmonics and the frequency intervals are determined by the cyclotron frequencies and the maximum frequency considered.
    - For two-species plasmas, it is assumed that the electron cyclotron frequency is an integer multiple of the ion cyclotron frequency.
    """

    # Define the dispersion relation 
    def D(w):
        be = 0.5 * (k * vthe / wce)**2
        bi = 0.5 * (k * vthi / wci)**2
        if field == 'Ez':
            # Formula (7.107) from Fitzpatrick
            if simulation[0] == 'fully kinetic':
                s = 1.0 - (c*k/w)**2
            elif simulation[0] == 'quasi-neutral':
                s = - (c*k/w)**2
            else:
                s= 0.0
                raise ValueError("Invalid simulation type. Choose 'fully kinetic' or 'quasi-neutral'.")
                    
            if simulation[2] == 'one species':
                # n = 0 contribution
                s = s - wpe**2 * np.exp(-be) * iv(0,be) / w**2
                for n in range(1,500):
                    s = s - wpe**2 * np.exp(-be) * 2 * iv(n,be) / (w**2 - n**2 * wce**2)
            elif simulation[2] == 'two species':
                # n = 0 contribution
                s = s - ( wpe**2 * np.exp(-be) * iv(0,be) / w**2
                        + wpi**2 * np.exp(-bi) * iv(0,bi)/ w**2)
                for n in range(1,50):
                    s = s - ( wpe**2 * np.exp(-be) * 2 * iv(n,be) / (w**2 - (n * wce)**2)  
                            + wpi**2 * np.exp(-bi) * 2 * iv(n,bi) / (w**2 - (n * wci)**2)) 
            return s
        elif field == 'Ex':
            # Formula (7.111) from Fitzpatrick (assumes c k/w >> 1)
            if simulation[0] == 'fully kinetic':
                s = 1.0 
            elif simulation[0] == 'quasi-neutral':
                s = 0.0
            else:
                s= 0.0
                raise ValueError("Invalid simulation type. Choose 'fully kinetic' or 'quasi-neutral'.")

            if simulation[2] == 'one species':
                for n in range(1,50):
                    s = s - wpe**2 / (w**2 - n**2 * wce**2) * 2 * n**2/be * iv(n,be) * np.exp(-be)
            elif simulation[2] == 'two species':
                for n in range(1,50):
                    s = s - ( wpe**2 / (w**2 - (n * wce)**2) * 2 * n**2/be * iv(n,be) * np.exp(-be) 
                            + wpi**2 / (w**2 - (n * wci)**2) * 2 * n**2/bi * iv(n,bi) * np.exp(-bi) ) 
            return s
   
    omega_max = 6  # maximum frequency to consider
    if simulation[2] == 'one species':
        n_omega = int(omega_max/wce)  # number of Bernstein wave frequencies
        w_sing = wce
    else:
        # we assume that wce is a integer multiple of wci which is the case if the mass ratio is an integer
        n_omega = int(omega_max/wci)  
        w_sing = wci
    Lmax = len(kk)  
    # Calculate the Bernstein wave dispersion relation
    roots_bernstein = np.zeros((n_omega,Lmax))
    eps = 1.e-14
    for i in range(1,Lmax):
        k = kk[i]
        for j in range (n_omega):
            try:
                roots_bernstein[j,i] = brentq(D, w_sing*(j)+eps, w_sing*(j+1)-eps)
            except(ValueError):
                roots_bernstein[j,i] = 0
    return roots_bernstein

plot_bernstein = False
# Plot the Bernstein wave dispersion relation for debugging purposes
if plot_bernstein:
    kk = np.linspace(0, 6, 100)  # wave numbers
    roots_bernstein = compute_roots_bernstein('Ex',kk)
    n_omega = roots_bernstein.shape[0]
    # for j in range (n_omega):
    #     print("roots_bernstein = ", np.max(roots_bernstein[j,:]))      
    #Plot the Bernstein wave dispersion relation
    plt.figure(figsize=(10, 6))
    plt.ylim(0, 6)
    for i in range(n_omega):
            plt.plot(kk,roots_bernstein[i,:],'g.',markersize=1)
    plt.title('Bernstein Wave Dispersion Relation')
    plt.xlabel('k')
    plt.ylabel('Frequency (w)')  
    plt.show()   


# %%
# This is for comparing the dispersion relation with the simulation

def plot_dispersion_relation(arr,field,HPDR=True):

    # Load the time series
    # The time series is created with the CreateSpaceTimeArrays.py script
    # make sure you are in the correct directory 
    print("Current working directory:", os.getcwd())
    ts = yt.load('./FullDiagnostics/plt_field??????')
    
    ntz = len(ts) # number of items in time series

    # Load the dataset to get the time and domain dimensions
    ds = ts[-1] # last time step
    time = ds.current_time
    x_left = np.array(ds.domain_left_edge)
    x_right = np.array(ds.domain_right_edge)
    L = x_right - x_left
    Lx = x_right[0] - x_left[0]

    arr = arr[:,:ntz]
    # apply the hann filter
    hann = np.hanning(ntz)
    arr = arr * hann
    # FFT in space and time
    arrfft = np.fft.fftn(arr)
    # normalize and transpose FFT array for plots
    arrfftnorm = np.transpose(np.abs(arrfft))/np.abs(arrfft).max()
    [N,L] = arrfftnorm.shape 

    T = time # last current time

    # define frequency (om) and wave number (kx) values
    Lmax = int(L/2)
    Nmax = int(N/2)
    om = 2*np.pi/T * np.arange(Nmax)[1:]
    kk = 2*np.pi/Lx * np.arange(Lmax)[1:]
    n_omega = 0
    roots_bernstein = np.zeros((1,1))
    roots = np.zeros((1,1))
    if HPDR:
        print("Roots of the hot plasma dispersion relation will be computed")
        if simulation[1] == 'parallel':
            roots = compute_HPDR(simulation,field,kk)
        elif simulation[1] == 'perpendicular':
            if field == 'Ex':
                # Bernstein wave dispersion relation
                roots_bernstein = compute_roots_bernstein('Ex',kk)
                n_omega = roots_bernstein.shape[0]
            elif field == 'Ez':
                # Bernstein wave dispersion relation 
                roots_bernstein = compute_roots_bernstein('Ez',kk)
                n_omega = roots_bernstein.shape[0]
            else:
                print("Field " + field + " not supported")
        # Cold Plasma dispersion relation along kx
    else:
        print("Roots of the cold plasma dispersion relation will be computed")
        roots = compute_CPDR(simulation, field, kk)
        n_omega = 0

    # levels for the contour plot
    lvls = np.logspace(-6, -0.5, 30)

    plt.figure(figsize=(10, 6))
    # plot the numerical dispersion relation
    plt.contourf(kk, om, arrfftnorm [1:Nmax, 1:Lmax], cmap="plasma",norm=colors.LogNorm(), levels=lvls, extend="both")
    plt.xlim([0, 6])
    plt.ylim([0, 6])
    
    # plot analytical solutions of dispersion relation
    plt.xlabel(r"$k_x c/\omega_{pe}$")
    plt.ylabel(r"Frequency ($\omega/\omega_{ce}$)")    
    if simulation[1] == 'perpendicular':
        if field == 'Ex':
            for i in range(n_omega):
                plt.plot(kk,roots_bernstein[i,:],'g.',markersize=1)
            plt.title('Dispersion relation from simulation (' + simulation[1] + ' Ex contour plot)')
        elif field == 'Ez':
            for i in range(n_omega):
                plt.plot(kk,roots_bernstein[i,:],'g.',markersize=1)           
            plt.title('Dispersion relation from simulation (' + simulation[1] + ' Ez contour plot)')
        else:
            print("Field " + field + " not supported")
    elif simulation[1] == 'parallel':
        if field == 'Ex':
            # Hot Plasma dispersion relation along kx (= kpar)
            plt.plot(kk, np.real(roots), color = 'black', ls ='dotted')
            plt.title('Dispersion relation from simulation (Ex contour plot)')
        elif field == 'Ez':
            # no wave in the QN case
            plt.plot(kk, np.real(roots), 'black', ls ='dotted') 
            plt.title('Dispersion relation from simulation (Ez contour plot)')
        else:
            print("Field not supported")   
    plt.show()



# %%
# A time-space array needs to be created by averaging over the y and z dimensions
# This is done with the CreateSpaceTimeArrays.py script

# Load the arrays from the file
field = 'Ex'
try:
    arr = np.load("t_x_array" + field + ".npy")
except FileNotFoundError:
    print(f"File t_x_array{field}.npy not found. Please run CreateSpaceTimeArrays.py first.")
    arr = None
if arr is not None:
    # Plot the dispersion relation for Ex
    print("Plotting dispersion relation for Ex")
plot_dispersion_relation(arr,field)

field = 'Ez'
try:
    arr = np.load("t_x_array" + field + ".npy")
except FileNotFoundError:
    print(f"File t_x_array{field}.npy not found. Please run CreateSpaceTimeArrays.py first.")
    arr = None
if arr is not None:
    # Plot the dispersion relation for Ez
    print("Plotting dispersion relation for Ez")
plot_dispersion_relation(arr,field)

# %%
# Check also the reduced diagnostics

import pandas as pd

# read electric energy
tabE=pd.read_csv("ReducedDiagnostics/ElecEnergy.txt",sep=r'\s+')
time = tabE.values[:,1]
ex2 = tabE.values[:,2]
ey2 = tabE.values[:,3]
ez2 = tabE.values[:,4]
etot = ex2+ey2+ez2
# read magnetic energy
try:
    tabB=pd.read_csv("ReducedDiagnostics/MagEnergy.txt",sep=r'\s+')
    tB = tabB.values[:,1]
    bx2 = tabB.values[:,2]
    by2 = tabB.values[:,3]
    bz2 = tabB.values[:,4]
except:
    print('MagEnergy.txt not found')
    bx2 = np.zeros_like(time)
    by2 = np.zeros_like(time)
    bz2 = np.zeros_like(time)
btot = bx2+by2+bz2
# read particle diagnostics
try:
    tabPart=pd.read_csv("ReducedDiagnostics/Part.txt",sep=r'\s+')
    tPart = tabPart.values[:,1]
    px = tabPart.values[:,2]
    py = tabPart.values[:,3]
    pz = tabPart.values[:,4]
    # read kinetic energy
    ekin = tabPart.values[:,5]
except:
    print('Part.txt not found')
    tPart = time
    px = np.zeros_like(time)
    py = np.zeros_like(time)
    pz = np.zeros_like(time)
    ekin = np.zeros_like(time)


# read error on Gauss law
try:
    tabGauss=pd.read_csv("ReducedDiagnostics/GaussError.txt",sep=r'\s+')
    tGauss = tabGauss.values[:,1]
    gaussError = tabGauss.values[:,2]
except:
    print('Gauss.txt not found')
    tGauss = time
    gaussError = np.zeros_like(time)

# plots
fig, axs = plt.subplots(3, 2,sharex=True,tight_layout=True)
axs[0,0].plot(time,ex2+ey2+ez2)
axs[0,0].set_title('electric energy')
axs[1,0].plot(time,bx2+by2+bz2)
axs[1,0].set_title('magnetic energy')
axs[0,1].plot(tPart,ekin)
axs[0,1].set_title('kinetic energy')
axs[1,1].plot(time,ex2+ey2+ez2+bx2+by2+bz2+ekin)
axs[1,1].set_title('total energy')
axs[2,0].plot(time,px)
axs[2,0].plot(time,py)
axs[2,0].plot(time,pz)
# label for particle momentum at the top
#axs[2,0].legend(['px','py','pz'])
axs[2,0].set_title('total particle momentum')
# remove initial time step
axs[2,1].plot(time[1:],gaussError[1:])
axs[2,1].set_title('Error on Gauss Law')
plt.show()

# %%
