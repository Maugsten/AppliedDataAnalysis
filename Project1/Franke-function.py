
# from calendar import c
from operator import index
from numpy import full
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from IPython import embed


def FrankeFunction(x, y):
    """
    Description:
        An implementation of the Franke function.
    Input:
        x (numpy array): Mesh with x values.
        y (numpy array): Mesh with y values.
    Output:
        z (numpy array): Mesh with function values.
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    z = term1 + term2 + term3 + term4
    return z


def create_X(x, y, n):
    """
    Description:
        Generates a design matrix X of polynomials with two dimensions.
        The code is loaned from the lectures notes (REFERENCE THIS).
    Input:
        x (numpy array): Mesh for x values.
        y (numpy array): Mesh for y values.
    Output:
        x (numpy array): Design matrix.
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)  # Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:, q+k] = (x**(i-k))*(y**k)

    return X


def svd_algorithm(X, z):
    """
    Description:
        Uses SVD-decomposition of the design matrix and the 
        Moore-Penrose pseudoinverse to calculate the optimal 
        parameters beta. Also uses SVD to calculate the variance 
        of each parameter.
    Input: 
        X (numpy array): Design matrix
        z (numpy array): 1D-array for the data we want to fit
    Return:
        betas (numpy array): Optimal parameters beta
        betas_variance (numpy array): Variance of each parameters beta
    """
    
    U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)
    betas = Vt.transpose() @ np.diag(1/Sigma) @ U.transpose() @ z
    betas_variance = sigma**2 * np.linalg.pinv(Vt.transpose() @ np.diag(Sigma) @ np.diag(Sigma) @ Vt)
    betas_variance = np.diag(betas_variance)

    return betas, betas_variance


def ordinary_least_squares(x, y, z, polydeg=5, resampling='None'):
    """
    Description:
        Preforms an ordinary least squares regression on the data z. 
        We assume that the data z might depend on x and y.
        The features in the design matrix are polynomials of x and y.
        The OLS regression can be done with bootstraping or cross-validation. 
    Input:
        x (numpy array): Mesh with x values.
        y (numpy array): Mesh with y values.
        z (numpy array): Mesh with z values. This is the data we try to make a fit for.
        polydeg (int): Highest order of polynomial degree in the design matrix.
        resampling (string): 
            Use 'None' for OLS without resampling.
            Use 'Bootstrap' for OLS with bootstrapping.
            Use 'CrossValidation' for OLS with cross-validation.
    Output:
        Plots
    """

    # Format data
    z_ = z.flatten().reshape(-1, 1)
    x_ = x.flatten()
    y_ = y.flatten()

    # Set the first order of polynomials for design matrix to be looped from
    startdeg = 1

    # Make arrays for MSEs and R2 scores
    MSE_train = np.zeros(polydeg-startdeg+1)
    MSE_test = np.zeros(polydeg-startdeg+1)
    R2_train = np.zeros(polydeg-startdeg+1)
    R2_test = np.zeros(polydeg-startdeg+1)

    # Make lists for parameters and variance
    collected_betas = []
    collected_betas_variance = []

    # Loop for OLS with increasing order of polynomial fit
    for i in range(startdeg, polydeg+1):

        # Make design matrix
        X = create_X(x, y, i)

        # Split into train and test data
        X_train, X_test, z_train, z_test = train_test_split(X, z_, test_size=0.2)

        if resampling == "Bootstrap":
            beta = np.zeros((len(X_train[0]), 1000)) # 1000 is number of bootstraps

            for j in range(1000): # For each bootstrap
                # Pick new randomly sampled training data and matching design matrix
                indx = np.random.randint(0,100,1000)
                X_train_btstrp = X_train[indx]
                z_train_btstrp = z_train[indx]

                # Calculate betas for this bootstrap
                beta[:,j] = svd_algorithm(X_train_btstrp, z_train_btstrp)[0].flatten()  

            # Average parameters over the bootstraps
            betas_averaged = np.mean(beta, axis=1)
            betas = np.zeros((len(X_train[0]),1))
            betas[:,0] = betas_averaged

            # Calculate variances of the parameters
            betas_variance = np.std(beta, axis=1)

        else:
            # OLS with SVD
            betas, betas_variance = svd_algorithm(X_train, z_train)

        # Calculate training fit and test fit
        z_tilde_train = X_train @ betas
        z_tilde_test = X_test @ betas

        # Collect parameters
        collected_betas.append(betas)
        collected_betas_variance.append(betas_variance)

        # Calculate mean square errors
        MSE_train[i-startdeg] = np.sum((z_train - z_tilde_train)**2)/len(z_train)
        MSE_test[i-startdeg] = np.sum((z_test - z_tilde_test)**2)/len(z_test)

        # Calculate mean
        z_mean = np.sum(z_)/len(z_)

        # Calculate R2 score
        R2_train[i-startdeg] = 1 - np.sum((z_train - z_tilde_train)**2)/np.sum((z_train - z_mean)**2)
        R2_test[i-startdeg] = 1 - np.sum((z_test - z_tilde_test)**2)/np.sum((z_test - z_mean)**2)

    # Calculate predicted values using all data (X)
    z_tilde = X @ svd_algorithm(X, z_)[0]

    # Get the right shape for plotting
    z_ = z_.reshape((len(x), len(x)))
    z_tilde = z_tilde.reshape((len(x), len(x)))

    z_tilde_train = X @ betas

    # Plots of the surfaces.
    fig = plt.figure(figsize=plt.figaspect(0.5), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax1.plot_surface(x, y, z_, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
    surf1 = ax2.plot_surface(x, y, z_tilde, cmap=cm.coolwarm,
                             linewidth=0, antialiased=False)
    ax1.set_zlim(-0.10, 1.40)
    ax1.zaxis.set_major_locator(LinearLocator(10))
    ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax1.set_title("Original data")
    ax2.set_zlim(-0.10, 1.40)
    ax2.zaxis.set_major_locator(LinearLocator(10))
    ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax2.set_title("OLS fit")
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.show()

    # Plot of the MSEs
    x_axis = np.linspace(startdeg, polydeg, polydeg-startdeg+1)
    plt.plot(x_axis, MSE_train, label="Training Sample")
    plt.plot(x_axis, MSE_test, label="Test Sample")
    plt.title("MSE vs Complexity")
    plt.xlabel("Model Complexity")
    plt.ylabel("Mean Square Error")
    plt.legend()
    plt.show()

    # Plot of the R2 scores
    plt.plot(x_axis, R2_train, label="Training Sample")
    plt.plot(x_axis, R2_test, label="Test Sample")
    plt.title("R2 score vs Complexity")
    plt.xlabel("Model Complexity")
    plt.ylabel("R2 score")
    plt.legend()
    plt.show()

    # Plotting paramaters and variance against terms
    plt.figure(figsize=(6, 4))
    for i in range(len(collected_betas)):
        plt.errorbar(range(1, len(collected_betas[i])+1), collected_betas[i].flatten(
        ), collected_betas_variance[i], fmt='o', capsize=6)
    plt.legend(['Order 1', 'Order 2', 'Order 3', 'Order 4', 'Order 5'])
    plt.title('Optimal parameters beta')
    plt.ylabel('Beta value []')
    plt.xlabel('Term index')
    plt.show()


if __name__ == "__main__":
    np.random.seed(69)  # Set seed so results can be reproduced.

    # Define domain. No need to scale this data as it's already in the range (0,1)
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)

    sigma = .1  # Standard deviation of the noise
    
    # Franke function with stochastic noise
    z = FrankeFunction(x, y) + np.random.normal(0, sigma, x.shape)

    # Preform regression
    ordinary_least_squares(x, y, z, polydeg=5, resampling='None')
    ordinary_least_squares(x, y, z, polydeg=5, resampling='Bootstrap')


    """
    Terminal>>>python filename.py
    (Generates plots)
    """