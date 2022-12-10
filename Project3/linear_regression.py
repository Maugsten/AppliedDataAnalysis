import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = np.linspace(0,1,int(4e2))
x0 = .5  # position of Gauss curve peak
s2 = .1
y = np.exp(-(x-x0)**2/s2)

def create_design_matrix(x,n):

    N = len(x)
    X = np.zeros(shape=(N,n))
    # X[:,0] = np.ones(1)
    
    for i in range(n):
        X[:,i] = x**(i+1)

    return X

X = create_design_matrix(x,4)
reg = LinearRegression().fit(X,y)

yhat = X @ reg.coef_ + reg.intercept_
score = reg.score(X,y)  # R2 score

print(f"R2 score = {score}")

plt.plot(x,y)
plt.plot(x,yhat)
plt.show()