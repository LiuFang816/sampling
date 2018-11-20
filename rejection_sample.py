import numpy as np
import matplotlib
import matplotlib.pyplot as plt
def q_sample():
    return np.random.rand()*4.

def p(x):
    return 0.3*np.exp(-(x-0.3)**2)+0.7* np.exp(-(x-2.)**2/0.3)

def rejection(nsamples):
    M = 0.9
    samples = np.zeros(nsamples, dtype=float)
    count = 0
    for i in range(nsamples):
        accept = False
        while not accept:
            x = q_sample()
            u = np.random.rand() * M
            if u < p(x):
                accept = True
                samples[i] = x
            else:
                count+=1
    print("reject count: ", count)
    return samples

x = np.arange(0,4,0.01)
x2 = np.arange(-0.5,4.5,0.1)
realdata = 0.3*np.exp(-(x-0.3)**2) + 0.7* np.exp(-(x-2.)**2/0.3)
box = np.ones(len(x2))*0.75#0.8
box[:5] = 0
box[-5:] = 0
plt.plot(x,realdata,'g',lw=3)
plt.plot(x2,box,'r--',lw=3)

import time
t0=time.time()
samples = rejection(10000)
t1=time.time()
print("Time ",t1-t0)

plt.hist(samples,15,normed=1,fc='c')
plt.xlabel('x',fontsize=24)
plt.ylabel('p(x)',fontsize=24)
plt.axis([-0.5,4.5,0,1])
plt.show()