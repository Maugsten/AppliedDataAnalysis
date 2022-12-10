import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from mlxtend.evaluate import bias_variance_decomp
import matplotlib.pyplot as plt

x = np.linspace(0,1,int(4e2))
x0 = .5  # position of Gauss curve peak
s2 = .1
y = np.exp(-(x-x0)**2/s2)

def create_design_matrix(x,n):

    N = len(x)
    X = np.zeros(shape=(N,n))
    
    for i in range(n):
        X[:,i] = x**(i+1)

    return X

X = create_design_matrix(x,4)

X_train, X_test, y_train, y_test, x_train, x_test = train_test_split(X, y, x, test_size=0.2)

reg = LinearRegression().fit(X_train, y_train)

ypred_train = reg.predict(X_train)
R2_train = reg.score(X_train, y_train) 
mse_train = mean_squared_error(y_train, ypred_train)

mse_test, bias, var = bias_variance_decomp(LinearRegression(), X_train, y_train, X_test, y_test, loss='mse')

print(f"R2 train = {R2_train}\nMSE train = {mse_train}\n\nMSE = {mse_test}\nBias = {bias}\nVariance = {var}")

plt.scatter(x_train, y_train, label="real")
plt.scatter(x_train, ypred_train, label="predicted")
plt.legend()
plt.show()