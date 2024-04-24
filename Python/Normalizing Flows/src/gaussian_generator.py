#%%
import numpy as np
import matplotlib.pyplot as plt
import random

#%%

mean = (0, 0)
cov = [[1,0],[0,1]]

mu = 0
sigma = 1
mu_Y = .3
sigma_Y = 1.7

Matransfo = np.random.exponential(size=(2,2))

#%%

def gauss_2d(mu, sigma):
    x = random.gauss(mu, sigma)
    y = random.gauss(mu, sigma)
    return (x, y)

def generation(N=10**4):
    X = [random.gauss(mu, sigma) for i in range(N)]
    Y = [random.gauss(mu_Y, sigma_Y) for i in range(N)]
    X = np.array(X).transpose()
    Y = np.array(Y).transpose()
    return np.vstack((X, Y))

def transfo(L):
    n = L.shape[1]
    ret = []
    for i in range(n):
        ret.append(Matransfo@L[:, i])
    return np.array(ret).transpose()

#%%

L = generation()
print(L.shape)

R = transfo(L)
print(R.shape)
#%%

plt.plot(L[0,:], L[1,:], 'x')
plt.plot(R[0,:], R[1,:], 'o')
plt.axis('equal')
plt.show()

#%%

path_to_save_o = "/Users/alexandreviolleau/Desktop/Data/original.txt"
path_to_save_e = "/Users/alexandreviolleau/Desktop/Data/exploit.txt"
np.savetxt(path_to_save_o, L, fmt='%1.4e')
np.savetxt(path_to_save_e, R, fmt='%1.4e')