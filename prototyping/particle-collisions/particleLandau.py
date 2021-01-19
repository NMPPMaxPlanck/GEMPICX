import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import math
from numba import cuda

# Can I use vectors inside the cuda function?
# Or am I restricted to elementary operatrions
@cuda.jit
def cuda_eom(f,v,vp,w,dv):
    i=cuda.grid(1)
    Q_00=0.
    Q_01=0.
    Q_10=0.
    Q_11=0.
    u_0=0.
    u_1=0.
    usq=0.
    if i<v.shape[0]:
        dv_0=0.
        dv_1=0.
        for j in range(v.shape[0]): 
           if i!=j:
                u_0=(v[i,0]-v[j,0]+vp[i,0]-vp[j,0])/2.
                u_1=(v[i,1]-v[j,1]+vp[i,1]-vp[j,1])/2.0
                u_sq=u_0**2+u_1**2
                u_cb=16.0*math.sqrt(u_sq)**3 # note the factor 16 here
                Q_00=(u_sq-u_0*u_0)/u_cb
                Q_01=-u_0*u_1/u_cb
                Q_10=Q_01
                Q_11=(u_sq-u_1*u_1)/u_cb
                dv_0-=w[j]*(Q_00*(f[i,0]-f[j,0])+Q_01*(f[i,1]-f[j,1]))
                dv_1-=w[j]*(Q_10*(f[i,0]-f[j,0])+Q_11*(f[i,1]-f[j,1]))
        dv[i,0]=dv_0
        dv[i,1]=dv_1

# cuda version to compute the entropy gradient
@cuda.jit
def cuda_f(v,w,quad_xi,quad_w,eps,f):
    i=cuda.grid(1)
    quad_v_0=0.
    quad_v_1=0.
    u_sq=0.
    if i<v.shape[0]:
        fi_0=0.
        fi_1=0.
        for q in range(quad_w.shape[0]):
            quad_v_0=math.sqrt(2*eps[0])*quad_xi[q,0]+v[i,0]
            quad_v_1=math.sqrt(2*eps[0])*quad_xi[q,1]+v[i,1]
            lnsum=0.
            for k in range(v.shape[0]): # This loop likely the slow one
                u_sq=(v[k,0]-quad_v_0)**2+(v[k,1]-quad_v_1)**2
                lnsum+=w[k]*math.exp(-u_sq/(2*eps[0]))/(2*math.pi*eps[0])
            lnsum=math.log(lnsum) # NOTE TO SELF: consider the + 1 here
            fi_0+=(quad_v_0-v[i,0])/eps[0]*lnsum*quad_w[q]
            fi_1+=(quad_v_1-v[i,1])/eps[0]*lnsum*quad_w[q]
        f[i,0]=fi_0
        f[i,1]=fi_1

# cuda version to evaluate the distribution at the mesh nodes
@cuda.jit
def cuda_distribution(w,v,eps,vn,fn):
    i=cuda.grid(1)
    if i<vn.shape[0]:
        fni=0.
        vsq=0.
        for k in range(v.shape[0]):
            vsq=(v[k,0]-vn[i,0])**2+(v[k,1]-vn[i,1])**2
            fni+=w[k]*math.exp(-vsq/(2*eps[0]))/(2*math.pi*eps[0])
        fn[i]=fni

        
class ParticleLandau():

    def __init__(self,n):
        # GPU thread number
        self.nthreads=256
        # mesh parameters
        self.N=n**2
        self.L=10.0
        self.h=2.0*self.L/n
        self.eps=0.64*self.h**(1.98)
        # Gauss-Hermite quadrature weights and nodes
        self.quad_w=[]
        self.quad_xi=[]
        self.init_quad()
        # create the initial population on the mesh
        # helpers
        u1=np.array([-2.0, 1.0])
        u2=np.array([0.0, -1.0])
        v=np.linspace(-self.L,self.L,n+1)
        v=(v[1:]+v[:-1])/2
        vx,vy=np.meshgrid(v,v)
        # actual data
        self.v=np.array([vx.flatten(),
                         vy.flatten()]).transpose() # particle coordinates
        self.vprev=deepcopy(self.v) # helper for iterations
        self.vc=deepcopy(self.v) # the grid nodes
        self.w=np.zeros(self.N) # particle weights
        for i in range(self.N): # compute the weights
            self.w[i]=(np.exp(-0.5*(self.v[i]-u1).dot(self.v[i]-u1))
                       +np.exp(-0.5*(self.v[i]-u2).dot(
                           self.v[i]-u2)))/(4*np.pi)*self.h**2

    # Initialize the quadrature nodes and weights
    def init_quad(self):
        xi=np.array([-2.3506049736745,
	             -1.3358490740137,
                     -0.43607741192762,
                     0.43607741192762,
                     1.3358490740137,
                     2.3506049736745])
        w=np.array([0.0045300099055088,
                    0.15706732032286,
                    0.72462959522439,
                    0.72462959522439,
                    0.15706732032286,
                    0.0045300099055088])
        xix,xiy=np.meshgrid(xi,xi)
        self.quad_xi=np.array([xix.flatten(),xiy.flatten()]).transpose()
        self.quad_w=np.outer(w,w).flatten()/np.pi

    # evaluates the momentum
    def momentum(self):
        return self.w.dot(self.v)

    # evaluates the energy
    def energy(self):
        return 0.5*self.w.dot(self.v[:,0]**2+self.v[:,1]**2)
    
    # cuda version of the quadrature rule for
    # the entropy vector
    def fvec_cuda(self):
        threadsperblock=self.nthreads 
        blockspergrid=math.ceil(self.N/threadsperblock)
        quad_xi_gbl=cuda.to_device(self.quad_xi)
        quad_w_gbl=cuda.to_device(self.quad_w)
        v_gbl=cuda.to_device(self.v) # global mem for the particle positions
        eps_gbl=cuda.to_device(np.array([self.eps])) # global mem for rbf parameter
        w_gbl=cuda.to_device(self.w) # global mem for the particle weights
        f_gbl=cuda.device_array((self.N,2)) # global mem for storing f
        cuda_f[blockspergrid,threadsperblock](v_gbl,w_gbl,quad_xi_gbl,quad_w_gbl,eps_gbl,f_gbl)
        return f_gbl.copy_to_host()
        

    # cuda version of the time step
    def step_cuda(self,dt):
        f=self.fvec_cuda()
        threadsperblock=self.nthreads 
        blockspergrid=math.ceil(f.shape[0]/threadsperblock)
        
        f_gbl=cuda.to_device(f) # global mem for the entropy gradient
        dv_gbl=cuda.device_array(f.shape) # global mem for storing dv
        vp_gbl=cuda.to_device(self.vprev) # global mem for the tmp particle positions
        w_gbl=cuda.to_device(self.w) # global mem for the particle weights

        energyTarget=self.energy()
        energyCurrent=energyTarget*2
        tolerance=1.e-15
        niter=0
        maxiter=10
        while abs(energyTarget-energyCurrent)/energyTarget>tolerance and niter<maxiter:
            niter+=1
            #print('iter='+str(niter)+'/'+str(maxiter))
            v_gbl=cuda.to_device(self.v) # global mem for the particle positions
            cuda_eom[blockspergrid,threadsperblock](f_gbl,v_gbl,vp_gbl,w_gbl,dv_gbl)
            self.v=self.vprev+dt*dv_gbl.copy_to_host() # update the velocity
            energyCurrent=self.energy()
        self.vprev=self.v*1.0
        
        
        

    # distribution at the nodes
    # cuda version
    def distribution_cuda(self):
        threadsperblock=self.nthreads 
        blockspergrid=math.ceil(self.N/threadsperblock)
        v_gbl=cuda.to_device(self.v) # global mem for the particle positions
        w_gbl=cuda.to_device(self.w) # global mem for the particle weights
        vn_gbl=cuda.to_device(self.vc) # global mem for the mesh nodes
        eps_gbl=cuda.to_device(np.array([self.eps])) # global mem for rbf parameter
        fn_gbl=cuda.device_array(self.w.shape) # global mem for storing distribution
        cuda_distribution[blockspergrid,threadsperblock](w_gbl,v_gbl,eps_gbl,vn_gbl,fn_gbl)
        return fn_gbl.copy_to_host()

    
    # computes the entropy gradient using
    # the Gauss-Hermite quadrature
    def fvec(self):
        f=np.zeros([self.N,2])
        for i in range(self.N):
            fivec=np.zeros(2)
            for q in range(len(self.quad_w)):
                vq=self.v[i]+np.sqrt(2*self.eps)*self.quad_xi[q]
                v=self.v-vq
                vsq=v[:,0]**2+v[:,1]**2
                rbf=np.exp(-vsq/(2*self.eps))/(2*np.pi*self.eps)
                lnsum=np.log(rbf.dot(self.w))
                fivec+=(vq-self.v[i])/self.eps*lnsum*self.quad_w[q]
            f[i]=fivec*1.0
        return f


    # integrate with explicit euler step
    def step(self,dt):
        f=self.fvec()
        #print(f[:10,:])
        dv=np.zeros([self.N,2])
        for i in range(self.N):
            #if np.mod(i,100)==0:
            #    print('prt='+str(i)+'/'+str(self.N))
            for j in range(self.N):
                if i != j :
                    u=self.v[i]-self.v[j]
                    usq=u.dot(u)
                    Q=(usq*np.eye(2)-np.outer(u,u))/(16*usq**(1.5))
                    dv[i]-=self.w[j]*Q.dot(f[i]-f[j])
        self.v=self.v+dv*dt
        

    # for visualization on a mesh
    def distribution(self):
        fN=np.zeros(self.N)
        for c in range(self.N):
            v=self.v-self.vc[c]
            vsq=v[:,0]**2+v[:,1]**2                                                                               
            rbf=np.exp(-vsq/(2*self.eps))/(2*np.pi*self.eps)
            fN[c]=rbf.dot(self.w)
        return fN
