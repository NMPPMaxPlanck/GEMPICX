import numpy as np
import matplotlib.pyplot as plt
from particleLandau import ParticleLandau
import time

n=40
dt=1.0
nt=200
l=ParticleLandau(n)
record_steps=[0,9,29,59,119,199]
start=time.time()
for i in range(nt):
    if i in record_steps:
        fN=l.distribution_cuda()
        plt.cla()
        plt.contourf(l.vc[:,0].reshape(n,n),l.vc[:,1].reshape(n,n),fN.reshape(n,n),20)
        plt.draw()
        plt.savefig('step_'+str(i+1).zfill(3)+'.png')
        print('step: '+str(i)+' momentum: '+str(l.momentum()[0])+' '+str(l.momentum()[1])+' energy: '+str(l.energy()))
    #if np.mod(i,10)==0:
    #    print('step='+str(i+1)+'/'+str(nt))
    #    #print('momentum: '+str(l.momentum())+' energy: '+str(l.energy()))
    l.step_cuda(dt)
    

print('***************************************')
print('cuda: '+str(n**2)+' prt')
print('cuda: '+str(time.time()-start)+' (s)')
print('***************************************')


