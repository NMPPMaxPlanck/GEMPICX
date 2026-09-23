#%%
# The cold plasma FK model in 1D in Z with B_0 along z reads:
#   dE_x/dt = -dB_y/dz - J_x
#   dE_y/dt = dB_x/dz - J_y
#   dB_x/dt = dE_y/dz
#   dB_y/dt =  -dE_x/dz
#   dJ_x/dt = omega_pe^2 * E_x + omega_ce * J_y
#   dJ_y/dt = omega_pe^2 * E_y - omega_ce * J_x
# We discretize space with N grid points and periodic BCs using finite differences and a central difference scheme.
# We form the spatial discretization matrix A (size 6N x 6N) such that
# dU/dt = A U, where U = [E_x, E_y, B_x, B_y, J_x, J_y]^T
# The Jacobian is then A, and we compute its eigenvalues to analyze stability.
import numpy as np
import matplotlib.pyplot as plt

def cold_plasma_matrix_FK(N, dz, omega_pe, omega_ce):
    """
    Build the 6N x 6N matrix A for the 1D cold plasma FK model with periodic BCs.
    Ordering: [E_x, E_y, B_x, B_y, J_x, J_y], each block length N.
    Central difference for d/dz with periodic wrap.
    """
    # derivative operator D (central difference, periodic)
    D = np.zeros((N, N), dtype=float)
    inv2dz = 1.0 / (2.0 * dz)
    for i in range(N):
        D[i, (i + 1) % N] =  inv2dz
        D[i, (i - 1) % N] = -inv2dz

    I = np.eye(N, dtype=float)
    Z = np.zeros((N, N), dtype=float)

    # allocate full matrix
    A = np.zeros((6 * N, 6 * N), dtype=float)

    def block_place(ar, br, block):
        A[ar * N:(ar + 1) * N, br * N:(br + 1) * N] = block

    # E_x' = -dB_y/dz - J_x
    block_place(0, 3, -D)   # couples to B_y
    block_place(0, 4, -I)   # couples to J_x

    # E_y' =  dB_x/dz - J_y
    block_place(1, 2, D)    # couples to B_x
    block_place(1, 5, -I)   # couples to J_y

    # B_x' = dE_y/dz
    block_place(2, 1, D)

    # B_y' = -dE_x/dz
    block_place(3, 0, -D)

    # J_x' = omega_pe^2 * E_x + omega_ce * J_y
    block_place(4, 0, (omega_pe ** 2) * I)
    block_place(4, 5, omega_ce * I)

    # J_y' = omega_pe^2 * E_y - omega_ce * J_x
    block_place(5, 1, (omega_pe ** 2) * I)
    block_place(5, 4, -omega_ce * I)

    return A

def cold_plasma_matrix_QN_woCurlB(N, dz, omega_pe, omega_ce, omega_pi, omega_ci):
    """
    QN cold plasma model without z-derivatives of B in the E-closure:
    E_x = coef * J_y, E_y = -coef * J_x
    Variables ordering: [B_x, B_y, J_x, J_y] (each length N).
    """
    if omega_pe == 0:
        raise ValueError("omega_pe must be nonzero for the QN closure")

    coef = omega_ce / (omega_pe ** 2)
    inv_dz = 1.0 / dz
    I = np.eye(N, dtype=float)

    # maps node quantities (J) -> half-grid derivative (for B')
    D_e2b = (np.roll(I, -1, axis=1) - I) * inv_dz

    Z = np.zeros((N, N), dtype=float)
    A = np.zeros((4 * N, 4 * N), dtype=float)

    def place(ar, br, M):
        A[ar * N:(ar + 1) * N, br * N:(br + 1) * N] = M

    # Ordering: [B_x, B_y, J_x, J_y]

    # B_x' = -coef * dJ_x/dz
    place(0, 0, Z)
    place(0, 1, Z)
    place(0, 2, -coef * D_e2b)
    place(0, 3, Z)

    # B_y' = -coef * dJ_y/dz
    place(1, 0, Z)
    place(1, 1, Z)
    place(1, 2, Z)
    place(1, 3, -coef * D_e2b)

    # J_x' = (omega_pi^2 * coef + omega_ci) * J_y
    place(2, 0, Z)
    place(2, 1, Z)
    place(2, 2, Z)
    place(2, 3, (omega_pi ** 2) * coef * I + omega_ci * I)

    # J_y' = - (omega_pi^2 * coef + omega_ci) * J_x
    place(3, 0, Z)
    place(3, 1, Z)
    place(3, 2, -((omega_pi ** 2) * coef + omega_ci) * I)
    place(3, 3, Z)

    return A

# Helper function to place blocks in a big matrix
def block_place(A, ar, br, block):
    N = block.shape[0]
    A[ar * N:(ar + 1) * N, br * N:(br + 1) * N] = block

# The cold plasma FKi-DKe model in 1D in z with B_0 along z reads:
#   dE_x/dt = (-c^2*dB_y/dz - J_x - (wpe^2 / wce) * E_y) / (1 + wpe^2 / wce^2)
#   dE_y/dt = (c^2*dB_x/dz - J_y + (wpe^2 / wce) * E_x) / (1 + wpe^2 / wce^2)
#   dB_x/dt = dE_y/dz
#   dB_y/dt =  -dE_x/dz
#   dJ_x/dt = omega_pi^2 * E_x + omega_ci * J_y
#   dJ_y/dt = omega_pi^2 * E_y - omega_ci * J_x

def cold_plasma_matrix(N, dz, omega_pe, omega_ce, omega_pi, omega_ci, c=1.0):
    """
    Staggered centered discretization (E,J on primary grid, B on half-grid).
    Returns 6N x 6N matrix A for the FKi-DKe model.
    J is scale to include 1/epsilon_0 factor.
    Variables ordering: [E_x, E_y, B_x, B_y, J_x, J_y] (each length N).
    Uses a Yee-like staggering: B on half-grid, E and J on nodes; derivatives use periodic wrap.

    Parameters:
    - omega_pe, omega_ce : electron plasma and cyclotron frequencies (entering E equations)
    - omega_pi, omega_ci : ion plasma and cyclotron frequencies (entering J equations)
    - c : speed of light (default 1.0); the E-equations include c^2 * curl(B)
    """

    inv_dz = 1.0 / dz
    I = np.eye(N, dtype=float)

    # D_b2e: maps B (at half-grid index j+1/2 stored as length-N array b_j)
    # to derivative at nodes i: (B_i - B_{i-1})/dz
    D_b2e = np.zeros((N, N), dtype=float)
    for i in range(N):
        D_b2e[i, i] =  inv_dz
        D_b2e[i, (i - 1) % N] = -inv_dz

    # D_e2b: maps E (at nodes) to derivative at half-grid j+1/2: (E_{j+1} - E_j)/dz
    D_e2b = np.zeros((N, N), dtype=float)
    for j in range(N):
        D_e2b[j, j] = -inv_dz
        D_e2b[j, (j + 1) % N] =  inv_dz

    # coefficients for modified E equations (electron parameters)
    denom = 1.0 + (omega_pe ** 2) / (omega_ce ** 2)
    alpha = (omega_pe ** 2) / omega_ce  # wpe^2 / wce

    A = np.zeros((6 * N, 6 * N), dtype=float)

    # E_x' = (-c^2 * dB_y/dz - J_x - alpha * E_y) / denom
    block_place(0, 3, -(c ** 2) * D_b2e / denom)         # couples to B_y
    block_place(0, 4, -I / denom)                        # couples to J_x
    block_place(0, 1, -(alpha / denom) * I)              # couples to E_y

    # E_y' = ( c^2 * dB_x/dz - J_y + alpha * E_x) / denom
    block_place(1, 2,  (c ** 2) * D_b2e / denom)         # couples to B_x
    block_place(1, 5, -I / denom)                        # couples to J_y
    block_place(1, 0,  (alpha / denom) * I)              # couples to E_x

    # B_x' = dE_y/dz  (half-grid derivative)
    block_place(2, 1, D_e2b)

    # B_y' = -dE_x/dz
    block_place(3, 0, -D_e2b)

    # J_x' = omega_pi^2 * E_x + omega_ci * J_y  (ion parameters)
    block_place(4, 0, (omega_pi ** 2) * I)
    block_place(4, 5, omega_ci * I)

    # J_y' = omega_pi^2 * E_y - omega_ci * J_x
    block_place(5, 1, (omega_pi ** 2) * I)
    block_place(5, 4, -omega_ci * I)

    return A
    
# The QN cold plasma FKi-DKe model in 1D in z with B_0 along z reads:
#   E_x = (-c**2*dB_x/dz + J_y ) * (wce/wpe^2)  
#   E_y = (-c**2*dB_y/dz - J_x) * (wce/wpe^2)
#   dB_x/dt = dE_y/dz
#   dB_y/dt =  -dE_x/dz
#   dJ_x/dt = omega_pi^2 * E_x + omega_ci * J_y
#   dJ_y/dt = omega_pi^2 * E_y - omega_ci * J_x
def cold_plasma_matrix_QN(N, dz, omega_pe, omega_ce, omega_pi, omega_ci):
    """
    Build the 4N x 4N matrix for the QN cold plasma model after eliminating E:
    variables ordering: [B_x, B_y, J_x, J_y] (each length N).
    Uses a Yee-like staggering: B on half-grid, J on nodes; derivatives use periodic wrap.
    """

    if omega_pe == 0:
        raise ValueError("omega_pe must be nonzero for the QN closure")

    coef = omega_ce / (omega_pe ** 2)   # E = coef * ( - dB/dz + J )
    inv_dz = 1.0 / dz
    I = np.eye(N, dtype=float)

    # D_b2e: maps B (half-grid stored length-N) -> derivative at nodes: (B_i - B_{i-1})/dz
    D_b2e = (I - np.roll(I, 1, axis=1)) * inv_dz

    # D_e2b: maps E (nodes) -> derivative at half-grid: (E_{j+1} - E_j)/dz
    D_e2b = (np.roll(I, -1, axis=1) - I) * inv_dz

    # Precompute compositions used below
    
    Lap_b = D_e2b.dot(D_b2e)  # maps B (half-grid) -> half-grid via node derivative in between
    D_e2b_mat = D_e2b      # reuse name for clarity

    A = np.zeros((4 * N, 4 * N), dtype=float)

    # Blocks for ordering [B_x, B_y, J_x, J_y]

    # B_x' = -coef * ( D_e2b @ D_b2e ) * B_y   +   -coef * D_e2b * J_x
    block_place(A, 0, 1, -coef * Lap_b)             # B_x <- B_y
    block_place(A, 0, 2, -coef * D_e2b_mat)         # B_x <- J_x


    # B_y' =  coef * ( D_e2b @ D_b2e ) * B_x   +   -coef * D_e2b * J_y
    block_place(A, 1, 0,  coef * Lap_b)             # B_y <- B_x
    block_place(A, 1, 3, -coef * D_e2b_mat)

    # J_x' = -omega_pi^2 * coef * D_b2e * B_x   +   (omega_pi^2 * coef + omega_ci) * J_y
    block_place(A, 2, 0, - (omega_pi ** 2) * coef * D_b2e)
    block_place(A, 2, 3, (omega_pi ** 2) * coef * I + omega_ci * I)

    # J_y' = -omega_pi^2 * coef * D_b2e * B_y   +   ( -omega_pi^2 * coef - omega_ci ) * J_x
    block_place(A, 3, 1, - (omega_pi ** 2) * coef * D_b2e)
    #block_place(A, 3, 2, - ( (omega_pi ** 2) * coef + omega_ci ) * I)
    return A

#%%
# example parameters
N = 128  # number of grid points
L = 1.0
dz = L / N
omega_pe = 1.0
omega_ce = -1.0
mass_ratio = 100.0  # mi/me
omega_pi = omega_pe / np.sqrt(mass_ratio)
omega_ci = -omega_ce / mass_ratio
A = cold_plasma_matrix_QN(N, dz, omega_pe, omega_ce, omega_pi,omega_ci)

coef = omega_ce / (omega_pe ** 2)
print((omega_pi ** 2) * coef + omega_ci)

# compute eigenvalues
eigs = dz*np.linalg.eigvals(A)

# diagnostics
print("max real(eig) =", np.max(eigs.real))
print("min real(eig) =", np.min(eigs.real))
print("max imag(eig) =", np.max(eigs.imag))
print("min imag(eig) =", np.min(eigs.imag))
cfl = 1/np.max(np.abs(eigs.imag))
print("CFL (max |imag(eig)|) =", cfl)
print("light speed", 1/np.sqrt(1+(omega_pe**2)/(omega_ce**2)))
print("wce/wpe^2 * cfl", -omega_ce**2/(omega_pe**2) *cfl)

# plot spectrum
plt.figure(figsize=(6, 6))
plt.scatter(eigs.real, eigs.imag, s=6)
plt.axvline(0, color="k", lw=0.5)
plt.xlabel("Re(eigenvalue)")
plt.ylabel("Im(eigenvalue)")
plt.title(f"Spectrum (N={N}, omega_pe={omega_pe}, omega_ce={omega_ce})")
plt.grid(True, ls=":", lw=0.5)
plt.tight_layout()
plt.show()

k = np.linspace(0, np.pi/dz, N//2)
omega_disp = - 0.5 * ( (omega_ce / (omega_pe ** 2)) * k )*(k + np.sqrt(k**2 + (4*omega_pi ** 2) ) )

plt.figure(figsize=(6,4))
plt.plot(k, omega_disp, 'b-', linewidth=2.5)
# %%
# Solve the differential equation dU/dt = A U with a RK3 method
def rk3_step(U, A, dt):
    """SSP RK3 step for dU/dt = A U."""
    k1 = A.dot(U)
    u1 = U + dt * k1
    k2 = A.dot(u1)
    u2 = 0.75 * U + 0.25 * (u1 + dt * k2)
    k3 = A.dot(u2)
    return (1.0 / 3.0) * U + (2.0 / 3.0) * (u2 + dt * k3)


# initial condition: small sinusoidal perturbation in E_x
k_mode = 1
amp = 1.0
z_grid = np.arange(N) / N 
E_x0 = amp * np.sin(2.0 * np.pi * k_mode * z_grid)
E_y0 = np.zeros(N)
B_x0 = np.zeros(N)
B_y0 = k_mode * amp * np.sin(2.0 * np.pi * k_mode * z_grid)
J_x0 = np.zeros(N)
J_y0 = np.zeros(N)

#U = np.concatenate([E_x0, E_y0, B_x0, B_y0, J_x0, J_y0])
U = np.concatenate([B_x0, B_y0, J_x0, J_y0])

# timestep and time window
dt = 1.7 * cfl * dz # small enough for stability (sqrt(3)/max|imag(eig)| for RK3)
t_final = 5
nsteps = int(np.ceil(t_final / dt))

# diagnostics storage
save_every = max(1, nsteps // 200)
times = []
energy = []

def total_energy(U):
    # simple L2 energy of all components
    # Note that energy should be conserved in the continuous model 
    # The contribution of the current to energy is divited by omega_pe^2 
    ii = 0
    E_x = np.zeros(N)
    E_y = np.zeros(N)
    if U.shape[0] % 6 != 0:
        E_x = U[0:N]
        E_y = U[N:2*N]
        ii = 2
    B_x = U[ii*N:(1+ii)*N]
    B_y = U[(1+ii)*N:(2+ii)*N]
    J_x = U[(2+ii)*N:(3+ii)*N]
    J_y = U[(3+ii)*N:(4+ii)*N]
    energy = 0.5 * np.sum(E_x.real ** 2 + E_y.real ** 2 + B_x.real ** 2 + B_y.real ** 2)
    if omega_pe == 0:
        return energy
    else:   
        return  energy + 0.5 / (omega_pe ** 2) * np.sum(J_x.real ** 2 + J_y.real ** 2)

t = 0.0
for istep in range(nsteps):
    U = rk3_step(U, A, dt)
    t += dt
    if istep % save_every == 0 or istep == nsteps - 1:
        times.append(t)
        energy.append(total_energy(U))
# plot the fields at final time
if U.shape[0] % 6 != 0:
    N = U.shape[0] // 4
    B_x_final = U[0:N]
    B_y_final = U[N:2*N]
    E_x_final = np.zeros(N)
    E_y_final = np.zeros(N)
else:
    E_x_final = U[0:N]
    E_y_final = U[N:2*N]
    B_x_final = U[2*N:3*N]
    B_y_final = U[3*N:4*N]
plt.figure(figsize=(8, 6))
z = np.arange(N) * dz
plt.plot(z, E_x_final.real, label="E_x")
plt.plot(z, E_y_final.real, label="E_y")
plt.plot(z, B_x_final.real, label="B_x")
plt.plot(z, B_y_final.real, label="B_y")
# plot exact solution for Maxwell as a check (Jx=Jy=0)
# E_x_exact = amp * np.sin(2.0 * np.pi * k_mode * (z-t))
# B_y_exact = amp * np.sin(2.0 * np.pi * k_mode * (z-t))
# plt.plot(z, E_x_exact, "k--", label="E_x exact")
# plt.plot(z, B_y_exact, "k:", label="B_y exact")
plt.xlabel("z")
plt.ylabel("Field values")
plt.title(f"Fields at t={t:.2f}")
plt.legend()
plt.grid(True, ls=":", lw=0.5)
plt.tight_layout()

# Compute error for pure Maxwell. Not valid with plasma
# error_E_x = np.linalg.norm(E_x_final - E_x_exact) / np.linalg.norm(E_x_exact)
# error_B_y = np.linalg.norm(B_y_final - B_y_exact) / np.linalg.norm(B_y_exact)
# print(f"Relative error in E_x: {error_E_x:.3e}")
# print(f"Relative error in B_y: {error_B_y:.3e}")

# plot energy vs time
plt.figure(figsize=(6, 3.5))
plt.plot(times, energy, "-o", ms=3)
plt.xlabel("t")
plt.ylabel("Total energy")
plt.title("Energy evolution (RK3)")
plt.grid(True, ls=":", lw=0.5)
plt.tight_layout()
plt.show()


# %%
q = 1.6e-19  # electron charge
me = 9.11e-31  # electron mass
epsilon_0 = 8.85e-12  # vacuum permittivity
n = 1e20  # electron number density (m^-3)
B0 = 3  # magnetic field strength (T)
wpe = np.sqrt(q**2 * n / (me * epsilon_0))
wce = q * B0 / me
print("Physical parameters:")
print(f"Plasma frequency wpe: {wpe:.3e} rad/s")
print(f"Cyclotron frequency wce: {wce:.3e} rad/s")
print(f"Ratio wpe/wce: {wpe/wce:.3e}")
# %%# %%
import numpy as np
import matplotlib.pyplot as plt

# Constants (chosen for simplicity)
a = 1.0
omega_p = 1.0

# Wavenumber range (k > 0)
k = np.linspace(0, 5, 500)

# Dispersion relations (physical positive-frequency solutions)
omega1 = (a * k / 2) * (k + np.sqrt(k**2 + 4 * omega_p))  # Upper branch
omega2 = (a * k / 2) * (-k + np.sqrt(k**2 + 4 * omega_p))  # Lower branch

# Plot
plt.figure(figsize=(10, 6))
plt.plot(k, omega1, 'b-', linewidth=2.5, label=r'$\omega_1 = \frac{a k}{2} \left[ k + \sqrt{k^2 + 4 \omega_p} \right]$')
plt.plot(k, omega2, 'r--', linewidth=2.5, label=r'$\omega_2 = \frac{a k}{2} \left[ -k + \sqrt{k^2 + 4 \omega_p} \right]$')

# Labels and title
plt.xlabel(r'$k$', fontsize=14)
plt.ylabel(r'$\omega$', fontsize=14)
plt.title(r'Dispersion Relation: $\omega(k)$', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.xlim(0, 5)
plt.ylim(0, 15)

# Asymptotic behavior annotation
plt.annotate(r'$\omega_1 \sim a k^2$ (quadratic)', xy=(4, 12), xytext=(3.5, 14),
             arrowprops=dict(arrowstyle='->', color='blue'))
plt.annotate(r'$\omega_2 \to a$ (constant)', xy=(4, 1), xytext=(3.5, 1.5),
             arrowprops=dict(arrowstyle='->', color='red'))

plt.show()
# %%
# Dispersion relation for the cold plasma QN model
import numpy as np
import matplotlib.pyplot as plt

# Constants
a = 1.0
omega_pi = 1.0
c = 1.0

# Wavenumber range (k > 0)
k = np.linspace(0.1, 5, 500)  # Avoid k=0 for numerical stability

# Dispersion relation
omega = (a * k / np.sqrt(2)) * np.sqrt(k**2 + np.sqrt(k**4 + 4 * omega_pi**2 * c**4))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(k, omega, 'b-', linewidth=2.5, label=r'$\omega = \frac{a k}{\sqrt{2}} \sqrt{k^2 + \sqrt{k^4 + 4 \omega_{pi}^2 c^4}}$')

# Labels and title
plt.xlabel(r'$k$', fontsize=14)
plt.ylabel(r'$\omega$', fontsize=14)
plt.title(r'Dispersion Relation: $\omega(k)$', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.xlim(0, 5)
plt.ylim(0, 15)

# Asymptotic behavior annotations
plt.annotate(r'$\omega \sim a \omega_{pi} c^2 k$ (linear)', xy=(1, 1.5), xytext=(0.5, 2.5),
             arrowprops=dict(arrowstyle='->', color='blue'))
plt.annotate(r'$\omega \sim a k^2$ (quadratic)', xy=(4, 12), xytext=(3.5, 14),
             arrowprops=dict(arrowstyle='->', color='blue'))

plt.show()
# %%
