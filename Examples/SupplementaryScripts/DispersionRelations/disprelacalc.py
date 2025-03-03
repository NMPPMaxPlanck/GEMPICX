import numpy as np
from numpy.polynomial.polynomial import Polynomial

def CPDR_calc(kc_0, theta_0, m_i, wp_e):
    """
    Calculates the cold plasma dispersion surfaces.
    
    Args:
        kc_0 (array): Array of k_perpendicular*c/w_c values.
        theta_0 (array): Array of theta angles in degrees.
        m_i (float): Ion mass in terms of electron masses.
        wp_e (float): Electron plasma frequency in terms of electron gyro frequency.
        
    Returns:
        np.ndarray: Sorted roots of the dispersion relation for each kc and theta.
    """
    # Convert single values to arrays if needed
    kc_0 = np.asarray(kc_0)
    theta_0 = np.asarray(theta_0)
    
    # If scalar, convert to 1D array
    if kc_0.ndim == 0:
        kc_0 = np.array([kc_0])
    if theta_0.ndim == 0:
        theta_0 = np.array([theta_0])
        
    # Convert theta from degrees to radians
    theta_0 = np.radians(theta_0)
    
    # Create 2D grids for kc and theta
    kc, theta = np.meshgrid(kc_0, theta_0, indexing='ij')
    
    # Ion and plasma frequencies
    wci = 1 / m_i  # Ion gyro frequency
    wp_i = wp_e / np.sqrt(m_i)  # Ion plasma frequency
    wp = np.sqrt(wp_e**2 + wp_i**2)  # Total plasma frequency
    
    # Precompute constants to speed up calculations
    kc2 = kc**2
    kc4 = kc2**2
    wci2 = wci**2
    wp2 = wp**2
    wextra = wp2 + wci  # Assume wce = 1.0
    wextra2 = wextra**2
    cos2theta = np.cos(theta)**2
    
    # Polynomial coefficients for the dispersion relation
    polkoeff8 = -(2 * kc2 + 1 + wci2 + 3 * wp2)
    polkoeff6 = kc4 + (2 * kc2 + wp2) * (1 + wci2 + 2 * wp2) + wextra2
    polkoeff4 = -(kc4 * (1 + wci2 + wp2) + 2 * kc2 * wextra2 +
                  kc2 * wp2 * (1 + wci2 - wci) * (1 + cos2theta) + wp2 * wextra2)
    polkoeff2 = (kc4 * (wp2 * (1 + wci2 - wci) * cos2theta + wci * wextra) +
                 kc2 * wp2 * wci * wextra * (1 + cos2theta))
    polkoeff0 = -kc4 * wci2 * wp2 * cos2theta
    
    # Initialize the output array
    num_roots = 10
    wfinal = np.zeros((num_roots, len(kc_0), len(theta_0)))
    
    # Solve the polynomial equation for each (kc, theta) pair
    for ktheta in range(len(theta_0)):
        for kkc in range(len(kc_0)):
            # Construct the polynomial coefficients
            disppolynomial = [
                1, 0, polkoeff8[kkc, ktheta], 0, polkoeff6[kkc, ktheta], 0,
                polkoeff4[kkc, ktheta], 0, polkoeff2[kkc, ktheta], 0,
                polkoeff0[kkc, ktheta]
            ]
            # Find the roots of the polynomial
            wtemp = np.roots(disppolynomial)
            # Sort the roots for consistent surfaces
            wfinal[:, kkc, ktheta] = np.sort(wtemp)
    
    return wfinal

def DeFi_calc(kc_0, theta_0, m_i, wp_e):
    """
    Calculates the Drift kinetic electron and Fully kinetic ioncold plasma dispersion surfaces.
    
    Args:
        kc_0 (array): Array of k_perpendicular*c/w_c values.
        theta_0 (array): Array of theta angles in degrees.
        m_i (float): Ion mass in terms of electron masses.
        wp_e (float): Electron plasma frequency in terms of electron gyro frequency.
        
    Returns:
        np.ndarray: Sorted roots of the dispersion relation for each kc and theta.
    """
    # Convert single values to arrays if needed
    kc_0 = np.asarray(kc_0)
    theta_0 = np.asarray(theta_0)
    
    # If scalar, convert to 1D array
    if kc_0.ndim == 0:
        kc_0 = np.array([kc_0])
    if theta_0.ndim == 0:
        theta_0 = np.array([theta_0])
        
    # Convert theta from degrees to radians
    theta_0 = np.radians(theta_0)
    
    # Create 2D grids for kc and theta
    kc, theta = np.meshgrid(kc_0, theta_0, indexing='ij')
    
    # Ion and plasma frequencies
    wci = 1 / m_i  # Ion gyro frequency
    wp_i = wp_e / np.sqrt(m_i)  # Ion plasma frequency
    wp = np.sqrt(wp_e**2 + wp_i**2)  # Total plasma frequency
    
    # Precompute constants to speed up calculations
    kc2 = kc**2
    kc4 = kc2**2
    wci2 = wci**2
    wp2 = wp**2
    wpe2 = wp_e**2
    wpi2 = wp_i**2
    cos2theta = np.cos(theta)**2
    wextra = wp2 + wci
    wextra2 = wextra**2

    # Polynomial coefficients for the dispersion relation
    polkoeff0 = -kc4 * wci2 * wp2 * cos2theta

    polkoeff2 = (kc4 * (wpe2 * (1 - wci2) * cos2theta +
                        wci * wextra ) +
                kc2 * wp2 * wci * wextra * (1 + cos2theta))

    polkoeff4 = -(kc4 * (1 + wpe2 * (1 - cos2theta) ) +
                kc2 * (wpe2 * (1 - wpi2 - wci2) - wpi2**2) * (1 + cos2theta) +
                2 * kc2 * wextra2 +
                wp2 * wextra2)

    polkoeff6 = (kc2 * (wpe2 + 1) * (2 + ( 1- cos2theta) * wpe2) +
                (wpe2**2 * (wci2 + wp2) +
                (wci2  + wpe2 + 3 * wpi2) +
                wpe2 * (2 * wci2 + 3 * wpe2 + 4 * wpi2)))

    polkoeff8 = -(1 + wpe2 )**2
        
    # Initialize the output array
    num_roots = 8
    wfinal = np.zeros((num_roots, len(kc_0), len(theta_0)))
    
    # Solve the polynomial equation for each (kc, theta) pair
    for ktheta in range(len(theta_0)):
        for kkc in range(len(kc_0)):
            # Construct the polynomial coefficients
            disppolynomial = [0, 0,
                polkoeff8, 0, polkoeff6[kkc, ktheta], 0,
                polkoeff4[kkc, ktheta], 0, polkoeff2[kkc, ktheta], 0,
                polkoeff0[kkc, ktheta]
            ]
            # Find the roots of the polynomial
            wtemp = np.roots(disppolynomial)
            # Sort the roots for consistent surfaces
            wfinal[:, kkc, ktheta] = np.sort(wtemp)
    
    return wfinal
