import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

x = np.random.rand(100)
y = 2.0+5*x*x+0.1*np.random.randn(100)

beta0 = np.ones(len(x))
beta1 = x
beta2 = x*x

M = np.array([beta0, beta1, beta2]).transpose()

weights = np.dot(np.dot(inv(np.dot(np.transpose(M), M)), np.transpose(M)), y)

long_x = np.linspace(0,1, int(1e3))
fit = weights[0] + weights[1]*long_x + weights[2]*long_x**2

print(weights)

plt.figure()
plt.scatter(x,y)
plt.plot(long_x,fit)
plt.legend(['Data', 'Fit'])
plt.show()