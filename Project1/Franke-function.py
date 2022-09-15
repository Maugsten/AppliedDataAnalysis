
from numpy import full
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from IPython import embed

# Make data. No need to scale this data as it's already in the range (0,1)
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

np.random.seed(1)

z = FrankeFunction(x, y) + np.random.normal(0, 0.1, x.shape)  # Franke function with stochastic noise

z_ = z.flatten().reshape(-1,1)
x_ = x.flatten()
y_ = y.flatten()

def create_X(x, y, n):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)  # Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X

startdeg = 1
polydeg = 5

MSE_train = np.zeros(polydeg-startdeg+1)
MSE_test = np.zeros(polydeg-startdeg+1)
R2_train = np.zeros(polydeg-startdeg+1)
R2_test = np.zeros(polydeg-startdeg+1)

 # SVD
def singular_value_decomp(X, z):
    """
    Input: 
        * X - design matrix
        * z_ - data we want to fit

    Return:
        * z_tilde - predicted values
        * betas - weights to apply on test data
    """
    U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)
    z_tilde = U @ U.transpose() @ z  
    betas =  np.linalg.pinv(X.transpose() @ X) @ X.transpose() @ z  # this is not the SVD way

    return z_tilde, betas

for i in range(startdeg,polydeg+1):

    # design matrix
    X = create_X(x,y,i)  

    # splitting into train and test data
    X_train, X_test, z_train, z_test, x_train, x_test, y_train, y_test = train_test_split(X,z_,x_,y_,test_size=0.2)

    z_tilde_train, betas = singular_value_decomp(X_train, z_train)
    z_tilde_test = X_test @ betas

    # mean square error
    MSE_train[i-startdeg] = np.sum((z_train - z_tilde_train)**2)/len(z_train)
    MSE_test[i-startdeg] = np.sum((z_test - z_tilde_test)**2)/len(z_test)

    # mean
    z_mean = np.sum(z_)/len(z_)

    # R2 score
    R2_train[i-startdeg] = 1 - np.sum((z_train - z_tilde_train)**2)/np.sum((z_train - z_mean)**2)
    R2_test[i-startdeg] = 1 - np.sum((z_test - z_tilde_test)**2)/np.sum((z_test - z_mean)**2)

# predicted values using all data (X)
z_tilde = singular_value_decomp(X, z_)[0]

# getting the right shape for plotting
z_ = z_.reshape((20,20))
z_tilde = z_tilde.reshape((20,20))

fig = plt.figure(figsize=plt.figaspect(0.5), constrained_layout=True)
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

# Plot the surface.
surf = ax1.plot_surface(x, y, z_, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
surf1 = ax2.plot_surface(x, y, z_tilde, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
                    
# Customize the z axis.
ax1.set_zlim(-0.10, 1.40)
ax1.zaxis.set_major_locator(LinearLocator(10))
ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax1.set_title("Original data")

ax2.set_zlim(-0.10, 1.40)
ax2.zaxis.set_major_locator(LinearLocator(10))
ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax2.set_title("OLS fit")

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()

x_axis = np.linspace(startdeg,polydeg,polydeg-startdeg+1)

plt.plot(x_axis, MSE_train, label="Training Sample")
plt.plot(x_axis, MSE_test, label="Test Sample")
plt.title("MSE vs Complexity")
plt.xlabel("Model Complexity")
plt.ylabel("Mean Square Error")
plt.legend()
plt.show()

plt.plot(x_axis, R2_train, label="Training Sample")
plt.plot(x_axis, R2_test, label="Test Sample")
plt.title("R2 score vs Complexity")
plt.xlabel("Model Complexity")
plt.ylabel("R2 score")
plt.legend()
plt.show()
