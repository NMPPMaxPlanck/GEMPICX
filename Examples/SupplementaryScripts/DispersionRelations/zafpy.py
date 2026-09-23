import numpy as np
import sympy as sp
import mpmath as mp  # multiple precision arithmetic
from scipy.linalg import hankel,eigvals
from scipy.integrate import quad
mp.dps = 30

def Z(x:complex):
    """ Plasma dispersion function """
    return sp.sqrt(sp.pi)*sp.exp(-x**2)*(sp.I-sp.erfi(x))
    
def Ndefault(omega,k):
    return 0

class zafpy:
    def __init__(self,D,kmode,N=Ndefault,max_zeros=20):
        self.kmode = kmode
        self.zeros = []
        omega = sp.symbols('omega')
        self.D = sp.lambdify(omega,D(omega,kmode),'mpmath')
        self.D_abs = sp.lambdify(omega,sp.Abs(D(omega,kmode)),'mpmath')
        self.D_real = sp.lambdify(omega,sp.re(D(omega,kmode)))
        self.D_imag = sp.lambdify(omega,sp.im(D(omega,kmode)))
        self.D_prime = sp.lambdify(omega,sp.diff(D(omega,kmode),omega),'mpmath')
        self.max_zeros = max_zeros
        self.Dprime_over_D = sp.lambdify(omega,
                    sp.diff(D(omega,kmode),omega)/D(omega,kmode),'mpmath')
        self.D_over_Dprime = sp.lambdify(omega,
                          D(omega,kmode)/sp.diff(D(omega,kmode),omega),'mpmath')
        self.N_over_Dprime = sp.lambdify(omega,
                          N(omega,kmode)/sp.diff(D(omega,kmode),omega),'mpmath')
        self.N_over_D = sp.lambdify(omega,N(omega,kmode)/D(omega,kmode),'mpmath')
        self.N_over_D_s = sp.lambdify(omega,N(omega,kmode)/D(omega,kmode)+N(-omega,kmode)/D(-omega,kmode),'mpmath')
        self.N_over_D_a = sp.lambdify(omega,N(omega,kmode)/D(omega,kmode)-N(-omega,kmode)/D(-omega,kmode),'mpmath')
        

    def D_reim(self, z):        
        """
        Returns the real and imaginary part of D(z) as a tuple (real, imag)
        """
        return [self.D_real(z[0]+1j*z[1]), self.D_imag(z[0]+1j*z[1])]

    def count_zeros(self, z0, z1, z2, z3, tol=1.e-7):
        """ Count the number of zeros in the box defined by xmin,xmax, ymin, ymax
            returns the number of zeros
        """
        k=self.kmode

        s1r,err1r=quad(lambda t: float((self.Dprime_over_D((1-t)*z0 + t*z1)*(z1-z0)).real),0,1,epsrel=tol)
        s1i,err1i=quad(lambda t: float((self.Dprime_over_D((1-t)*z0 + t*z1)*(z1-z0)).imag),0,1,epsrel=tol)
        s2r,err2r=quad(lambda t: float((self.Dprime_over_D((1-t)*z1 + t*z2)*(z2-z1)).real),0,1,epsrel=tol)
        s2i,err2i=quad(lambda t: float((self.Dprime_over_D((1-t)*z1 + t*z2)*(z2-z1)).imag),0,1,epsrel=tol)
        s3r,err3r=quad(lambda t: float((self.Dprime_over_D((1-t)*z2 + t*z3)*(z3-z2)).real),0,1,epsrel=tol)
        s3i,err3i=quad(lambda t: float((self.Dprime_over_D((1-t)*z2 + t*z3)*(z3-z2)).imag),0,1,epsrel=tol)
        s4r,err4r=quad(lambda t: float((self.Dprime_over_D((1-t)*z3 + t*z0)*(z0-z3)).real),0,1,epsrel=tol)
        s4i,err4i=quad(lambda t: float((self.Dprime_over_D((1-t)*z3 + t*z0)*(z0-z3)).imag),0,1,epsrel=tol)
        
        assert(s1r+s2r+s3r+s4r<tol)
        return (s1i + s2i + s3i + s4i)/(2*np.pi)
        
        
    def get_zeros(self, z0, z1, z2, z3, deg=3, tol=1e-12, tolK=0.01, maxiter=10, verbose=False):
        # count zeros in box
        nzeros = self.count_zeros(z0, z1, z2, z3)
        K=round(nzeros)
        if mp.fabs(K-nzeros.real) > tolK or K>5:
            if verbose:
                print('refining: error=', mp.fabs(K-nzeros.real), ' K=', K) 
            self.refine(z0, z1, z2, z3, deg, tol, tolK, maxiter, verbose)
        else:
            if verbose:
                print('found ', K, ' zeros, Error = ', mp.fabs(K-nzeros))
            # Compute s_m if K>0
            if K>0:
                s=np.zeros(2*K,'complex')
                for m in range(0,2*K):
                    s1=mp.quad(lambda t: ((1-t)*z0 + t*z1)**m/self.D((1-t)*z0 + t*z1)*(z1-z0),[0,1],maxdegree=deg)/(2j*mp.pi)
                    s2=mp.quad(lambda t: ((1-t)*z1 + t*z2)**m/self.D((1-t)*z1 + t*z2)*(z2-z1),[0,1],maxdegree=deg)/(2j*mp.pi)
                    s3=mp.quad(lambda t: ((1-t)*z2 + t*z3)**m/self.D((1-t)*z2 + t*z3)*(z3-z2),[0,1],maxdegree=deg)/(2j*mp.pi)
                    s4=mp.quad(lambda t: ((1-t)*z3 + t*z0)**m/self.D((1-t)*z3 + t*z0)*(z0-z3),[0,1],maxdegree=deg)/(2j*mp.pi)
                    s[m]= s1+s2+s3+s4
                    
                # Compute zeros as generalised eigenvalues of Hankel matrices
                H = hankel(s[0:K], s[K-1:2*K-1])
                H2 = hankel(s[1:K+1], s[K:2*K])
                w = eigvals(H2,H)
                # Check Error on zero and perform Newton refinement if necessary
                error_flag = False
                for i, ww in enumerate(w):
                    error = mp.fabs(self.D(ww))
                    it = 0
                    while error > tol and it<maxiter:
                        ww=ww-self.D_over_Dprime(ww)
                        error = mp.fabs(self.D(ww))
                        it += 1
                    w[i]=ww
                    if verbose:
                        print(str(ww)+': error on zero= ' + str(error) + ' #iter='+str(it))
                    if (error > tol):
                        error_flag = True
                        break
                if error_flag:
                    self.refine(z0, z1, z2, z3,deg,tol,tolK,maxiter,verbose)
                else:
                    self.zeros += list(w)
            
        return self.zeros
        
    def refine(self,z0, z1, z2, z3, deg, tol, tolK, maxiter, verbose):
        """ Get zeros in refined box """
        z01 = (z0+z1)/2
        z12 = (z1+z2)/2
        z23 = (z2+z3)/2
        z03 = (z0+z3)/2
        z_center =  (z0+z1+z2+z3)/4

        self.get_zeros(z0, z01, z_center, z03, deg, tol, tolK, maxiter, verbose)
        self.get_zeros(z01, z1, z12, z_center, deg, tol, tolK, maxiter, verbose)
        self.get_zeros(z12, z2, z23, z_center, deg, tol, tolK, maxiter, verbose)
        self.get_zeros(z23, z3, z03, z_center, deg, tol, tolK, maxiter, verbose)
    
    def compute_zeros_newton(self, x0, tol=1e-10, eps=1e-6, maxiter=100, verbose=0):
        """ Compute zeros using Newton's method """
        x = x0
        for i in range(maxiter):
            f = self.D(x)
            df = self.D_prime(x)
            if mp.fabs(f) < tol:
                return x
            if mp.fabs(df) > eps:
                x = x - f/df
                
            else: # Picards method
                print("Warning: derivative is too small, using Picard's method", df)
                x = x - f
            if verbose == 1:
                print(f"Iteration {i+1}: x = {x}, f(x) = {f}")
        raise ValueError("Newton's method did not converge after {} iterations".format(maxiter))
        return x
