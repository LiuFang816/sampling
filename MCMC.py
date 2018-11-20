# encoding: utf-8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn
mu = 3
sigma = 10
def qsample():
    return np.random.normal(mu,sigma)
def q(x):
    return np.exp(-(x-mu)**2/(sigma**2))
def p(x):
    """目标分布"""
    return 0.3*np.exp(-(x-0.3)**2) + 0.7* np.exp(-(x-2.)**2/0.3)
def hm(n=10000):
    sample = np.zeros(n)
    sample[0] = 0.5
    for i in range(n-1):
        q_s = qsample()
        u = np.random.rand()
        if u < min(1, (p(q_s)*q(sample[i]))/(p(sample[i])*q(q_s))):
            sample[i+1] = q_s
        else:
            sample[i+1] = sample[i]
    return sample

x = np.arange(0,4,0.1)
realdata = p(x)
N=10000
sample = hm(N)
plt.plot(x,realdata,'g',lw=3)
plt.plot(x,q(x),'r')
plt.hist(sample,bins=x,normed=1,fc='c')
plt.show()