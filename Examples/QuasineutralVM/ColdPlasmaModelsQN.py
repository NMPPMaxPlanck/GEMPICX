#%%
#   The QN cold plasma FKi-DKe model in 1D in z with B_0 along z reads:
#   E_x = (-dB_x/dz + J_y ) * (wce/wpe^2)  
#   E_y = (-dB_y/dz - J_x) * (wce/wpe^2)
#   dB_x/dt = dE_y/dz =(-d^2B_y/dz^2 - dJ_x/dz) * (wce/wpe^2)
#   dB_y/dt =  -dE_x/dz = (d^2B_x/dz^2 - dJ_y/dz ) * (wce/wpe^2) 
#   dJ_x/dt = omega_pi^2 * E_x + omega_ci * J_y = -(wce*wpi**2/wpe**2) dBx/dz +(wce*wpi**2/wpe**2+wci)Jy
#   dJ_y/dt = omega_pi^2 * E_y - omega_ci * J_x = -(wce*wpi**2/wpe**2) dBy/dz -(wce*wpi**2/wpe**2+wci)Jy
# Note that E_x and E_y are can be directly plugged into the 4 other equations.
# We discretize space with N grid points and periodic BCs using finite differences 
# and a staggered central difference scheme, J at nodes and B at cell centers
# We form the spatial discretization matrix A (size 4N x 4N) such that
# dU/dt = A U, where U = [B_x, B_y, J_x, J_y]^T
# The Jacobian matrix is then A, and we compute its eigenvalues to analyze stability.
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def cold_plasma_matrix_QN_FKi_DKe(N, dz, omega_pe, omega_ce, omega_pi, omega_ci):
    """
    Build the 4N x 4N matrix for the QN FKi-DKe cold plasma model after eliminating E:
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
    #Lap_b = np.zeros((N, N), dtype=float)
    #D_e2b_mat = D_e2b      # reuse name for clarity

    Z = np.zeros((N, N), dtype=float)

    A = np.zeros((4 * N, 4 * N), dtype=float)

    def place(ar, br, M):
        A[ar * N:(ar + 1) * N, br * N:(br + 1) * N] = M

    # Blocks for ordering [B_x, B_y, J_x, J_y]

    # B_x' = -coef * ( D_e2b @ D_b2e ) * B_y   +   -coef * D_e2b * J_x
    place(0, 0, Z)                          # B_x does not depend not on B_x directly
    place(0, 1, -coef * Lap_b)              # B_x <- B_y
    place(0, 2, -coef * D_e2b)              # B_x <- J_x
    place(0, 3, Z)                          # B_x <- J_y

    # B_y' =  coef * ( D_e2b @ D_b2e ) * B_x   +   -coef * D_e2b * J_y
    place(1, 0,  coef * Lap_b)             # B_y <- B_x
    place(1, 1, Z)
    place(1, 2, Z)
    place(1, 3, -coef * D_e2b)

    # J_x' = -omega_pi^2 * coef * D_b2e * B_x   +   (omega_pi^2 * coef + omega_ci) * J_y
    place(2, 0, - (omega_pi ** 2) * coef * D_b2e)
    place(2, 1, Z)
    place(2, 2, Z)
    place(2, 3, ((omega_pi ** 2) * coef + omega_ci) * I)

    # J_y' = -omega_pi^2 * coef * D_b2e * B_y   +   ( -omega_pi^2 * coef - omega_ci ) * J_x
    place(3, 0, Z)
    place(3, 1, - (omega_pi ** 2) * coef * D_b2e)
    place(3, 2, - ((omega_pi ** 2) * coef + omega_ci ) * I)
    place(3, 3, Z)

    return A

# For the semi-implicit scheme we remove curl B from the matrix that will
# be treated explicitly 
# The QN cold plasma FKi-DKe model in 1D in z with B_0 along z reads
# without the curl B contribution
#   E_x = J_y * (wce/wpe^2)  
#   E_y = - J_x * (wce/wpe^2)
#   dB_x/dt = dE_y/dz
#   dB_y/dt =  -dE_x/dz
#   dJ_x/dt = omega_pi^2 * E_x + omega_ci * J_y
#   dJ_y/dt = omega_pi^2 * E_y - omega_ci * J_x
def cold_plasma_matrix_QN_FKi_DKe_woCurlB(N, dz, omega_pe, omega_ce, omega_pi, omega_ci):
    """
    QN cold plasma model for FKi-DKe without z-derivatives of B in the E_perp expression:
    E_x = J_y * (wce/wpe^2), E_y = - J_x * (wce/wpe^2)
    Variables ordering: [B_x, B_y, J_x, J_y] (each length N).
    """
    if omega_pe == 0:
        raise ValueError("omega_pe must be nonzero for the QN closure")

    coef = omega_ce / (omega_pe ** 2)   # E = coef * ( - dB/dz + J )
    print("coef",(omega_pi ** 2) * coef + omega_ci) # should be 0
    inv_dz = 1.0 / dz
    I = np.eye(N, dtype=float)

    # D_b2e: maps B (half-grid stored length-N) -> derivative at nodes: (B_i - B_{i-1})/dz
    D_b2e = (I - np.roll(I, 1, axis=1)) * inv_dz

    # D_e2b: maps E (nodes) -> derivative at half-grid: (E_{j+1} - E_j)/dz
    D_e2b = (np.roll(I, -1, axis=1) - I) * inv_dz

    # Precompute compositions used below
    Lap_b = D_e2b.dot(D_b2e)  # maps B (half-grid) -> half-grid via node derivative in between

    Z = np.zeros((N, N), dtype=float)

    A = np.zeros((4 * N, 4 * N), dtype=float)

    def place(ar, br, M):
        A[ar * N:(ar + 1) * N, br * N:(br + 1) * N] = M

    # Blocks for ordering [B_x, B_y, J_x, J_y]

    # B_x' =   -coef * D_e2b * J_x
    place(0, 0, Z)              # B_x does not depend not on B_x directly
    place(0, 1, Z)              # B_x <- B_y
    place(0, 2, -coef * D_e2b)              # B_x <- J_x
    place(0, 3, Z)                          # B_x <- J_y

    # B_y' =    +   -coef * D_e2b * J_y
    place(1, 0,  Z)             # B_y <- B_x
    place(1, 1, Z)
    place(1, 2, Z)
    place(1, 3, -coef * D_e2b)

    # J_x' = -omega_pi^2 * coef * D_b2e * B_x   +   (omega_pi^2 * coef + omega_ci) * J_y
    place(2, 0, - (omega_pi ** 2) * coef * D_b2e)
    place(2, 1, Z)
    place(2, 2, Z)
    place(2, 3, ((omega_pi ** 2) * coef + omega_ci) * I)

    # J_y' = -omega_pi^2 * coef * D_b2e * B_y   +   ( -omega_pi^2 * coef - omega_ci ) * J_x
    place(3, 0, Z)
    place(3, 1, - (omega_pi ** 2) * coef * D_b2e)
    place(3, 2, - ((omega_pi ** 2) * coef + omega_ci ) * I)
    place(3, 3, Z)

    return A

def crank_nicolson_matrices(N, dz, dt, coef):
    """
    Assemble matrices M1 and M2 for Crank-Nicolson scheme:
    M1 U^{n+1} = M2 U^n
    where U = [B, C] with periodic BCs.
    
    PDE:
    dB/dt = - coef*(C_{i+1} - 2*C_i + C_{i-1})/dz^2
    dC/dt = coef*(B_{i+1} - 2*B_i + B_{i-1})/dz^2
    """
    inv_dz2 = 1.0 / (dz ** 2)
    I = np.eye(N, dtype=float)
    
    # Laplacian matrix: (u_{i+1} - 2*u_i + u_{i-1})/dz^2
    diag_main = -2.0 * np.ones(N)
    diag_plus = np.ones(N-1)
    diag_minus = np.ones(N-1)
    
    Lap = (np.diag(diag_plus, k=1) + np.diag(diag_main) + np.diag(diag_minus, k=-1)) * inv_dz2
    # Apply periodic BCs
    Lap[0, -1] = inv_dz2
    Lap[-1, 0] = inv_dz2
    
    Z = np.zeros((N, N), dtype=float)
    
    # M1 = I - (dt/2) * A, M2 = I + (dt/2) * A
    # where A = [[0, Lap], [-Lap, 0]]
    M1 = np.zeros((2*N, 2*N), dtype=float)
    M2 = np.zeros((2*N, 2*N), dtype=float)

    M1[0:N, 0:N] = I
    M1[0:N, N:2*N] = dt/2 * Lap * coef
    M1[N:2*N, 0:N] = -dt/2 * Lap * coef
    M1[N:2*N, N:2*N] = I
    
    M2[0:N, 0:N] = I
    M2[0:N, N:2*N] = -dt/2 * Lap * coef
    M2[N:2*N, 0:N] = dt/2 * Lap * coef
    M2[N:2*N, N:2*N] = I
    
    return M1, M2

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

def total_energy(U):
    # simple L2 energy of all components
    # Note that energy should be conserved in the continuous model 
    # The contribution of the current to energy is divited by omega_pe^2 
    B_x = U[0:N]
    B_y = U[N:2*N]
    J_x = U[2*N:3*N]
    J_y = U[3*N:4*N]
    energyB = 0.5 * np.sum(B_x.real ** 2 + B_y.real ** 2)
    energyJ = 0.5 / (omega_pi ** 2) * np.sum(J_x.real ** 2 + J_y.real ** 2)  
    return  energyB + energyJ

def crank_nicolson_step(U, N, dz, dt, coef):
    "Solve M1 B^n+1 = M2 C, where M1 and M2 define the Crank-Nicolson scheme "
    Bx = U[0:N]
    By = U[N:2*N]
    Bxk = sp.fft.fft(Bx)
    Byk = sp.fft.fft(By)
    K = np.arange(N) * 2*np.pi / N
    a = coef * dt/dz**2 * (np.cos(K) - 1)
    Bxknew = ((1 - a**2) * Bxk - 2 * a * Byk) / (1 + a**2)
    Byknew =  (2 * a * Bxk + (1 - a**2) * Byk) / (1 + a**2)
    U[0:N] = np.real(sp.fft.ifft(Bxknew))
    U[N:2*N] = np.real(sp.fft.ifft(Byknew))
    # M1, M2 = crank_nicolson_matrices(N,dz,dt,coef)
    # C = np.linalg.solve(M1, np.dot(M2,U[0:2*N]))
    # U[0:2*N] = C
    return U

#%%
# Determine stability criterion for explicit part:
# 1) for the full model
# 2) for the model without the curl B term that is treated implicitly
# example parameters
N = 128  # number of grid points
L = 1.0
dz = L / N
omega_pe = 50.0
omega_ce = -1.0
mass_ratio = 10.0  # mi/me
omega_pi = omega_pe / np.sqrt(mass_ratio)
omega_ci = np.abs(omega_ce) / mass_ratio
Afull = cold_plasma_matrix_QN_FKi_DKe(N, dz, omega_pe, omega_ce, omega_pi, omega_ci)
Aexp = cold_plasma_matrix_QN_FKi_DKe_woCurlB(N, dz, omega_pe, omega_ce, omega_pi, omega_ci)

print("wc/wp**2 ", dz/(omega_ce/omega_pe**2), dz/(omega_ci/omega_pi**2))


# compute eigenvalues
eigsFull = np.linalg.eigvals(Afull)
eigsExp = np.linalg.eigvals(Aexp)

# diagnostics
print("==== Eigenvalues of full system =====")
print("max real(eig) =", np.max(eigsFull.real))
print("min real(eig) =", np.min(eigsFull.real))
print("max imag(eig) =", np.max(eigsFull.imag))
print("min imag(eig) =", np.min(eigsFull.imag))
cflExp = np.sqrt(3)/np.max(np.abs(eigsFull.imag))/dz
print("CFL for RK3 (max |dt/dz|) =", cflExp)

# diagnostics
print("==== Eigenvalues of explicit part of system =====")
print("max real(eig) =", np.max(eigsExp.real))
print("min real(eig) =", np.min(eigsExp.real))
print("max imag(eig) =", np.max(eigsExp.imag))
print("min imag(eig) =", np.min(eigsExp.imag))
cflExpImp = np.sqrt(3)/np.max(np.abs(eigsExp.imag))/dz 
print("CFL = (max |dt/dz|) =", cflExpImp)

# plot spectrum
plt.figure(figsize=(6, 6))
#plt.scatter(eigsFull.real, eigsFull.imag, color="b", s=6)
plt.scatter(eigsExp.real, eigsExp.imag*dz, color="g", s=6)
plt.axvline(0, color="k", lw=0.5)
plt.xlabel("Re(eigenvalue)")
plt.ylabel("Im(eigenvalue)")
plt.title(f"Spectrum (N={N}, omega_pe={omega_pe}, omega_ce={omega_ce})")
plt.grid(True, ls=":", lw=0.5)
plt.tight_layout()
plt.show()
#%%
# Problem with 4 variables Bx, By, Jx, Jy
# initial condition: small sinusoidal perturbation in E_x
k_mode = 1
amp = 1.0
z = np.arange(N) * dz
B_x0 = k_mode * amp * np.cos(2.0 * np.pi * k_mode * z)
B_y0 = k_mode * amp * np.sin(2.0 * np.pi * k_mode * z)
J_x0 = np.zeros(N)
J_y0 = np.zeros(N)

U = np.concatenate([B_x0, B_y0, J_x0, J_y0])

## Fully explicit scheme
scheme = "implicit"
# timestep and time window
if scheme == "explicit":
    dt = 0.9 * cflExp * dz # small enough for stability (sqrt(3)/max|imag(eig)| for RK3)
else:
    dt = 1.9 * cflExpImp * dz # dtmax is multiplied by two as scheme is applied on dt/2
    #dt = 0.99 * cflExp * dz 

t_final = 50
nsteps = int(np.ceil(t_final / dt))

print("time:", dt, t_final, nsteps)

# diagnostics storage, start with initial condition
save_every = max(1, nsteps // 200)
times = [0.0]
energy = [total_energy(U)]

coef = omega_ce/omega_pe**2
t = 0.0
for istep in range(nsteps):
    if scheme == "explicit":
        U = rk3_step(U, Afull, dt)
    else:
        U = rk3_step(U, Aexp, dt/2)
        U = crank_nicolson_step(U, N, dz, dt, coef)
        U = rk3_step(U, Aexp, dt/2)
        
    #print(f"Step {istep+1}/{nsteps}, time {t+dt:.4f}")
    t += dt
    if istep % save_every == 0 or istep == nsteps - 1:
        times.append(t)
        energy.append(total_energy(U))
# plot the fields at final time
B_x_final = U[0:N]
B_y_final = U[N:2*N]
J_x_final = U[2*N:3*N]
J_y_final = U[3*N:4*N]
plt.figure(figsize=(8, 6))

plt.plot(z, B_x_final.real, label="B_x")
plt.plot(z, B_y_final.real, label="B_y")
plt.plot(z, J_x_final.real, label="J_x")
plt.plot(z, J_y_final.real, label="J_y")

# plot exact solution for Crank-Nicolson part only
# Bx_exact = k_mode * amp * np.cos(2.0 * np.pi * k_mode * (z - coef * 2.0 * np.pi * k_mode * t))
# By_exact = k_mode * amp * np.sin(2.0 * np.pi * k_mode * (z - coef * 2.0 * np.pi * k_mode * t))
# plt.plot(z, Bx_exact, "k--", label="Bx exact")
# plt.plot(z, By_exact, "k:", label="By exact")

plt.xlabel("z")
plt.ylabel("Field values")
plt.title(f"Fields at t={t:.2f}")
plt.legend()
plt.grid(True, ls=":", lw=0.5)
plt.tight_layout()

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
