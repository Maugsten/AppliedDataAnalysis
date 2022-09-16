
from numpy import full
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from IPython import embed

fig = plt.figure()
ax = fig.gca(projection='3d')

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

np.random.seed(10)

z = FrankeFunction(x, y) + np.random.normal(0, 0.1, x.shape)  # Franke function with stochastic noise

z_ = z.flatten().reshape(-1,1)
x_ = x.flatten()
y_ = y.flatten()

# Setting up the design matrix with fifth order polynomial
z0 = np.ones(len(x_))

# 1st order
z1 = x_ 
z2 = y_

# 2nd order
z3 = x_**2
z4 = y_**2
z5 = x_*y_

# 3rd order
z6 = x_**3
z7 = y_**3
z8 = x_**2 * y_
z9 = x_ * y_**3

# 4th order
z10 = x_**4
z11 = y_**4
z12 = x_**2 * y_**2
z13 = x_**3 * y_
z14 = x_ * y_**3

#5th order
z15 = x_**5
z16 = y_**5
z17 = x_**4 * y_
z18 = x_ * y_**4
z19 = x_**3 * y_**2
z20 = x_**2 * y_**3

list_of_features = [z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10,\
    z11, z12, z13, z14, z15, z16, z17, z18, z19, z20]

startdeg = 1
polydeg = 5

z_tilde_train = 0

MSE_train = np.zeros(polydeg-startdeg+1)
MSE_test = np.zeros(polydeg-startdeg+1)
R2_train = np.zeros(polydeg-startdeg+1)
R2_test = np.zeros(polydeg-startdeg+1)

for i in range(startdeg,polydeg+1):

    if startdeg < 1:
        raise ValueError("first polynomial degree must be 1 or higher")
    
    if i == 1:
        new_list_of_features = list_of_features[0:3]
    elif i == 2:
        new_list_of_features = list_of_features[0:6]
    elif i == 3:
        new_list_of_features = list_of_features[0:10]
    elif i == 4:
        new_list_of_features = list_of_features[0:15]
    elif i == 5:
        new_list_of_features = list_of_features[0:21]
    else:
        raise ValueError("this code only allows polynomial degrees up to fifth order")

    X = np.array(new_list_of_features).transpose()  # design matrix

    # splitting into train and test data
    X_train, X_test, z_train, z_test, x_train, x_test, y_train, y_test = train_test_split(X,z_,x_,y_,test_size=0.2)
    print(np.shape(x_train))
    
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

# get the right shape for plotting
z_tilde_train = z_tilde_train.reshape((-1,1))
z_tilde_test = z_tilde_test.reshape((-1,1))
# print(np.shape(z_tilde_train))

# Plot the surface.
surf = ax.plot_surface(x_train, y_train, z_train, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
surf1 = ax.plot_surface(x_train, y_train, z_tilde_train, cmap=cm.bone,
                    linewidth=0, antialiased=False)
                    
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf1, shrink=0.5, aspect=5)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
