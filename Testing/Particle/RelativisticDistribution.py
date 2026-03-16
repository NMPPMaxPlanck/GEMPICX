# %%
import numpy as np
from scipy.integrate import quad
from scipy.special import kn


# %% define marginal Maxwell-Juettner distribution function with relativistic shift
# see eq. (33) in "Loading relativistic Maxwell distributions in particle simulations", Zenitani 2015
def f_marginal(u, T, beta=0):
    gamma = 1 / np.sqrt(1 - beta**2)
    return (
        (gamma * np.sqrt(1 + u**2) + T)
        / (2 * gamma**3 * kn(2, 1 / T))
        * np.exp(-gamma * (np.sqrt(1 + u**2) - beta * u) / T)
    )


# %% compute moments of distribution function by adaptive quadrature to test sampler in GEMPICX
tol = 1e-10  # choose tolerance as low as possible
a = -100  # large domain for accurate integral
b = 100

# direction without shift
print(
    [
        quad(lambda u: u**moment * f_marginal(u, 1.1**2, 0), a, b, epsabs=tol, epsrel=tol)[0]
        for moment in range(4)
    ]
)

# direction with shift
print(
    [
        quad(
            lambda u: u**moment * f_marginal(u, 1.1**2, 0.5), a, b, epsabs=tol, epsrel=tol
        )[0]
        for moment in range(4)
    ]
)
