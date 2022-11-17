import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import KFold 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


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
        n (int): Highest polynomial degree
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

    return betas

def ridge_solver(X, z, lmd):
    """
    Description:

    Input: 
        X (numpy array): Design matrix
        z (numpy array): 1D-array for the data we want to fit
    Return:
        betas (numpy array): Optimal parameters beta
        betas_variance (numpy array): Variance of each parameters beta
    """
    XTX = X.transpose() @ X
    betas = np.linalg.pinv(XTX + lmd*np.eye(len(XTX))) @ X.transpose() @ z

    return betas

def LASSO_solver(X, z, lmd):
    """
    Description:

    Input: 
        X (numpy array): Design matrix
        z (numpy array): 1D-array for the data we want to fit
    Return:
        betas (numpy array): Optimal parameters beta
        betas_variance (numpy array): Variance of each parameters beta
    """
    RegLasso = linear_model.Lasso(lmd, fit_intercept=False)
    RegLasso.fit(X,z)
    betas = RegLasso.coef_.reshape(-1,1)

    return betas


def make_plots(x, y, z, z_, z_tilde, startdeg, polydeg, MSE_train, MSE_test, R2_train, R2_test, bias, vari, surface=False):
    """
    Description:
        Makes a lot of different plots
    """

    # get the correct shape to plot 3D figures
    m = np.shape(z)[0]
    n = np.shape(z)[1]

    x_scaled = (x - np.min(x))/(np.max(x) - np.min(x))
    y_scaled = (y - np.min(y))/(np.max(y) - np.min(y))

    z_plot = z_.reshape((m,n))
    z_tilde_plot = z_tilde.reshape((m,n))

    min_val = np.min(z_)
    max_val = np.max(z_)

    # heatmaps
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True) 
    im1 = ax1.imshow(z_plot, vmin=min_val, vmax=max_val)
    ax1.set_title("Original data")
    ax1.set_xlabel("nx")
    ax1.set_ylabel("ny")
    fig.colorbar(im1, ax=ax1, location='bottom')

    im2 = ax2.imshow(z_tilde_plot, vmin=min_val, vmax=max_val)
    ax2.set_title("OLS fit")
    ax2.set_xlabel("nx")
    ax2.set_ylabel("ny")
    fig.colorbar(im2, ax=ax2, location='bottom')
    
    im3 = ax3.imshow(z_tilde_plot - z_plot, vmin=min_val, vmax=max_val)
    ax3.set_title("Difference")
    ax3.set_xlabel("nx")
    ax3.set_ylabel("ny")
    fig.colorbar(im3, ax=ax3, location='bottom')

    if surface == True:
        fig = plt.figure(figsize=plt.figaspect(0.5), constrained_layout=True)
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')

        surf1 = ax1.plot_surface(x_scaled, y_scaled, z_plot, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)
        surf2 = ax2.plot_surface(x_scaled, y_scaled, z_tilde_plot, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)

        ax1.set_zlim(min_val, max_val)
        ax1.zaxis.set_major_locator(LinearLocator(10))
        ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.yaxis._axinfo['label']
        ax1.set_zlabel('z')
        ax1.set_title("Original data")

        ax2.set_zlim(min_val, max_val)
        ax2.zaxis.set_major_locator(LinearLocator(10))
        ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.yaxis._axinfo['label']
        ax2.set_zlabel('z')
        ax2.set_title("OLS fit")
        fig.colorbar(surf1, shrink=0.5, aspect=10)

    # Plot MSE
    x_axis = np.linspace(startdeg, polydeg, polydeg-startdeg+1)
    plt.figure(figsize=(6, 4))
    plt.plot(x_axis, MSE_train, '--.', label="Training Data")
    plt.plot(x_axis, MSE_test, '--.', label="Test Data")
    plt.title("MSE against Complexity")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Mean Square Error")
    plt.legend()

    # Plot the R2 scores
    plt.figure(figsize=(6, 4))
    plt.plot(x_axis, R2_train, '--.', label="Training Data")
    plt.plot(x_axis, R2_test, '--.', label="Test Data")
    plt.title("R2 score against Complexity")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("R2 score")
    plt.legend()

    # Plot the Bias-Variance of test data
    plt.figure(figsize=(6, 4))
    plt.plot(x_axis, np.log10(MSE_test), '--.', label="MSE")
    plt.plot(x_axis, np.log10(bias), '--.', label="Bias")
    plt.plot(x_axis, np.log10(vari), '--.', label="Variance")
    # plt.plot(x_axis, bias+vari, '--', label="sum")
    plt.title("Bias-Variance Tradeoff")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("log10(Error)")
    plt.legend()

    plt.show()

def ordinary_least_squares(x, y, z, polydeg=5, startdeg=1, resampling='None', k=10):
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

    # Make arrays for MSEs and R2 scores
    MSE_train = np.zeros(polydeg-startdeg+1)
    MSE_test = np.zeros(polydeg-startdeg+1)
    R2_train = np.zeros(polydeg-startdeg+1)
    R2_test = np.zeros(polydeg-startdeg+1)

    # Make lists for parameters and variance
    bias = np.zeros(polydeg-startdeg+1)
    vari = np.zeros(polydeg-startdeg+1)

    parameters = []

    # Loop for OLS with increasing order of polynomial fit
    for i in range(startdeg, polydeg+1):

        # Make design matrix
        X = create_X(x, y, i)
        X = X[:,1:]

        if resampling == "Bootstrap":
            # Split into train and test data
            X_train, X_test, z_train, z_test = train_test_split(X, z_, test_size=0.2)

            # scaling 
            for j in range(len(X_test[0,:])):
                X_test[:,j] = (X_test[:,j] - np.amin(X_train[:,j])) / (np.amax(X_train[:,j]) - np.amin(X_train[:,j]))   # Normalization
                X_train[:,j] = (X_train[:,j] - np.amin(X_train[:,j])) / (np.amax(X_train[:,j]) - np.amin(X_train[:,j])) # Normalization
            z_test = (z_test - np.mean(z_train))/np.std(z_train)   # Standardization
            z_train = (z_train - np.mean(z_train))/np.std(z_train) # Standardization
            
            n_bootstraps = 100

            z_tilde_train = np.zeros((len(X_train), n_bootstraps)) 
            z_tilde_test = np.zeros((len(X_test), n_bootstraps)) 

            for j in range(n_bootstraps): # For each bootstrap

                # Pick new randomly sampled training data and matching design matrix
                indx = np.random.randint(0,len(X_train),len(X_train))
                X_train_btstrp = X_train[indx]
                z_train_btstrp = z_train[indx]

                b = svd_algorithm(X_train_btstrp, z_train_btstrp)
                z_tilde_train[:,j] = X_train @ b.flatten()
                z_tilde_test[:,j] = X_test @ b.flatten()

            MSE_train[i-startdeg] = np.mean(np.mean((z_train - z_tilde_train)**2, axis=1, keepdims=True))
            MSE_test[i-startdeg] = np.mean(np.mean((z_test - z_tilde_test)**2, axis=1, keepdims=True))
            bias[i-startdeg] = np.mean((z_test - np.mean(z_tilde_test, axis=1, keepdims=True))**2)        
            vari[i-startdeg] = np.mean(np.var(z_tilde_test, axis=1, keepdims=True))
            R2_train[i-startdeg] = 1 - np.sum((z_train - np.mean(z_tilde_train, axis=1, keepdims=True))**2)/np.sum((z_train - np.mean(z_train))**2)
            R2_test[i-startdeg] = 1 - np.sum((z_test - np.mean(z_tilde_test, axis=1, keepdims=True))**2)/np.sum((z_test - np.mean(z_test))**2)

        elif resampling == "CrossValidation":
            # Truncate z_ so to be able to divide into equally sized k folds
            floor = len(z_) // k
            z_cv = z_[:floor*k]
            X_cv = X[:floor*k,:]

            shuffler = np.random.permutation(len(X_cv))
            X_shuffled = X_cv[shuffler]
            z_shuffled = z_cv[shuffler]

            X_groups = np.array_split(X_shuffled, k)
            z_groups = np.array_split(z_shuffled, k)

            z_tilde_train = np.zeros((len(X_cv)-len(X_groups[0]), k))
            z_tilde_test = np.zeros((len(X_groups[0]), k)) 

            mse_train = []
            mse_test = []
            r2_train = []
            r2_test = []
            bia = []
            var = []

            for j in range(k):                
                X_test = X_groups[j] # Picks j'th matrix as test matrix
                X_groups_copy = X_groups # Makes copy of the groups, to not mess up later
                X_groups_reduced = np.delete(X_groups_copy, j, 0) # Remove j'th matrix from the copy-group
                X_train = np.concatenate((X_groups_reduced), axis=0) # Concatenates the remainding matrices

                # Same for the data z 
                z_test = z_groups[j] 
                z_groups_copy = z_groups 
                z_groups_reduced = np.delete(z_groups_copy, j, 0)
                z_train = np.concatenate((z_groups_reduced), axis=0)

                # scaling 
                for f in range(len(X_test[0,:])):
                    X_test[:,f] = (X_test[:,f] - np.amin(X_train[:,f])) / (np.amax(X_train[:,f]) - np.amin(X_train[:,f]))   # Normalization
                    X_train[:,f] = (X_train[:,f] - np.amin(X_train[:,f])) / (np.amax(X_train[:,f]) - np.amin(X_train[:,f])) # Normalization
                z_test = (z_test - np.mean(z_train))/np.std(z_train)   # Standardization
                z_train = (z_train - np.mean(z_train))/np.std(z_train) # Standardization

                # OLS with SVD
                b = svd_algorithm(X_train, z_train).flatten()
                
                z_tilde_train[:,j] = X_train @ b
                z_tilde_test[:,j] = X_test @ b

                z_train = z_train.flatten()
                z_test = z_test.flatten()

                mse_train.append(np.mean((z_train - z_tilde_train[:,j])**2))
                mse_test.append(np.mean((z_test - z_tilde_test[:,j])**2))

                r2_train.append(1 - np.sum((z_train - z_tilde_train[:,j])**2)/np.sum((z_train - np.mean(z_train))**2))
                r2_test.append(1 - np.sum((z_test - z_tilde_test[:,j])**2)/np.sum((z_test - np.mean(z_test))**2))

                bia.append(np.mean( (z_test - np.mean(z_tilde_test[:,j]))**2 ))
                var.append(np.var(z_tilde_test[:,j]))

            MSE_train[i-startdeg] = np.mean(mse_train)
            MSE_test[i-startdeg] = np.mean(mse_test)
            R2_train[i-startdeg] = np.mean(r2_train)
            R2_test[i-startdeg] = np.mean(r2_test) 
            bias[i-startdeg] = np.mean(bia)        
            vari[i-startdeg] = np.mean(var)
            
        else:
            # Split into train and test data
            X_train, X_test, z_train, z_test = train_test_split(X, z_, test_size=0.2)

            # scaling 
            for j in range(len(X_test[0,:])):
                X_test[:,j] = (X_test[:,j] - np.amin(X_train[:,j])) / (np.amax(X_train[:,j]) - np.amin(X_train[:,j]))   # Normalization
                X_train[:,j] = (X_train[:,j] - np.amin(X_train[:,j])) / (np.amax(X_train[:,j]) - np.amin(X_train[:,j])) # Normalization
            z_test = (z_test - np.amin(z_train)) / (np.amax(z_train) - np.amin(z_train))   # Normalization
            z_train = (z_train - np.amin(z_train)) / (np.amax(z_train) - np.amin(z_train)) # Normalization
            """ LEGG MERKE TIL NORMALISERING HER! """
        
            # OLS with SVD
            betas = svd_algorithm(X_train, z_train)
            parameters.append(betas)

            z_tilde_train = np.zeros((len(X_train), 1))
            z_tilde_test = np.zeros((len(X_test), 1))
            
            z_tilde_train = X_train @ betas
            z_tilde_test = X_test @ betas 

            MSE_train[i-startdeg] = np.mean(np.mean((z_train - z_tilde_train)**2, axis=1, keepdims=True))
            MSE_test[i-startdeg] = np.mean(np.mean((z_test - z_tilde_test)**2, axis=1, keepdims=True))
            bias[i-startdeg] = np.mean( (z_test - np.mean(z_tilde_test, axis=1, keepdims=True))**2 )        
            vari[i-startdeg] = np.mean( np.var(z_tilde_test, axis=1, keepdims=True) )
            R2_train[i-startdeg] = 1 - np.sum((z_train - np.mean(z_tilde_train, axis=1, keepdims=True))**2)/np.sum((z_train - np.mean(z_train))**2) # z_?
            R2_test[i-startdeg] = 1 - np.sum((z_test - np.mean(z_tilde_test, axis=1, keepdims=True))**2)/np.sum((z_test - np.mean(z_test))**2) # z_?
    
    # Calculate predicted values using all data (X)
    # z_scaled = (z_ - np.mean(z_))/np.std(z_)
    # z_tilde = X @ svd_algorithm(X, z_scaled)
    z_tilde = X @ svd_algorithm(X, z_)

    make_plots(x,y,z,z_,z_tilde,startdeg,polydeg,MSE_train,MSE_test,R2_train,R2_test,bias,vari,surface=True)
    return MSE_test#, parameters


def ridge(x, y, z, lmd, polydeg=5, startdeg=1, resampling='None', k=10):
    """
    Description:
        Preforms a Ridge regression on the data z. 
        We assume that the data z might depend on x and y.
        The features in the design matrix are polynomials of x and y.
        The Ridge regression can be done with bootstraping or cross-validation. 
    Input:
        x (numpy array): Mesh with x values.
        y (numpy array): Mesh with y values.
        z (numpy array): Mesh with z values. This is the data we try to make a fit for.
        lmd (float): Hyperparameter for the Ridge regression.
        polydeg (int): Highest order of polynomial degree in the design matrix.
        resampling (string): 
            Use 'None' for Ridge without resampling.
            Use 'Bootstrap' for Ridge with bootstrapping.
            Use 'CrossValidation' for Ridge with cross-validation.
    Output:
        Plots
    """

    # Format data
    z_ = z.flatten().reshape(-1, 1)

    # Make arrays for MSEs and R2 scores
    MSE_train = np.zeros(polydeg-startdeg+1)
    MSE_test = np.zeros(polydeg-startdeg+1)
    R2_train = np.zeros(polydeg-startdeg+1)
    R2_test = np.zeros(polydeg-startdeg+1)

    # Make lists for parameters and variance
    bias = np.zeros(polydeg-startdeg+1)
    vari = np.zeros(polydeg-startdeg+1)

    parameters = []

    # Loop for OLS with increasing order of polynomial fit
    for i in range(startdeg, polydeg+1):

        # Make design matrix
        X = create_X(x, y, i)
        X = X[:,1:]

        if resampling == "Bootstrap":
            # Split into train and test data
            X_train, X_test, z_train, z_test = train_test_split(X, z_, test_size=0.2)

            # scaling 
            for j in range(len(X_test[0,:])):
                X_test[:,j] = (X_test[:,j] - np.amin(X_train[:,j])) / (np.amax(X_train[:,j]) - np.amin(X_train[:,j]))   # Normalization
                X_train[:,j] = (X_train[:,j] - np.amin(X_train[:,j])) / (np.amax(X_train[:,j]) - np.amin(X_train[:,j])) # Normalization
            z_test = (z_test - np.mean(z_train))/np.std(z_train)   # Standardization
            z_train = (z_train - np.mean(z_train))/np.std(z_train) # Standardization
            
            n_bootstraps = 100

            z_tilde_train = np.zeros((len(X_train), n_bootstraps)) 
            z_tilde_test = np.zeros((len(X_test), n_bootstraps)) 

            for j in range(n_bootstraps): # For each bootstrap

                # Pick new randomly sampled training data and matching design matrix
                indx = np.random.randint(0,len(X_train),len(X_train))
                X_train_btstrp = X_train[indx]
                z_train_btstrp = z_train[indx]

                b = ridge_solver(X_train_btstrp, z_train_btstrp, lmd)
                z_tilde_train[:,j] = X_train @ b.flatten()
                z_tilde_test[:,j] = X_test @ b.flatten()

            MSE_train[i-startdeg] = np.mean(np.mean((z_train - z_tilde_train)**2, axis=1, keepdims=True))
            MSE_test[i-startdeg] = np.mean(np.mean((z_test - z_tilde_test)**2, axis=1, keepdims=True))
            bias[i-startdeg] = np.mean((z_test - np.mean(z_tilde_test, axis=1, keepdims=True))**2)        
            vari[i-startdeg] = np.mean(np.var(z_tilde_test, axis=1, keepdims=True))
            R2_train[i-startdeg] = 1 - np.sum((z_train - np.mean(z_tilde_train, axis=1, keepdims=True))**2)/np.sum((z_train - np.mean(z_train))**2)
            R2_test[i-startdeg] = 1 - np.sum((z_test - np.mean(z_tilde_test, axis=1, keepdims=True))**2)/np.sum((z_test - np.mean(z_test))**2)

        elif resampling == "CrossValidation":
            # Truncate z_ so to be able to divide into equally sized k folds
            floor = len(z_) // k
            z_cv = z_[:floor*k]
            X_cv = X[:floor*k,:]

            shuffler = np.random.permutation(len(X_cv))
            X_shuffled = X_cv[shuffler]
            z_shuffled = z_cv[shuffler]

            X_groups = np.array_split(X_shuffled, k)
            z_groups = np.array_split(z_shuffled, k)

            z_tilde_train = np.zeros((len(X_cv)-len(X_groups[0]), k))
            z_tilde_test = np.zeros((len(X_groups[0]), k)) 

            mse_train = []
            mse_test = []
            r2_train = []
            r2_test = []
            bia = []
            var = []

            for j in range(k):                
                X_test = X_groups[j] # Picks j'th matrix as test matrix
                X_groups_copy = X_groups # Makes copy of the groups, to not mess up later
                X_groups_reduced = np.delete(X_groups_copy, j, 0) # Remove j'th matrix from the copy-group
                X_train = np.concatenate((X_groups_reduced), axis=0) # Concatenates the remainding matrices

                # Same for the data z 
                z_test = z_groups[j] 
                z_groups_copy = z_groups 
                z_groups_reduced = np.delete(z_groups_copy, j, 0)
                z_train = np.concatenate((z_groups_reduced), axis=0)

                # scaling 
                for f in range(len(X_test[0,:])):
                    X_test[:,f] = (X_test[:,f] - np.amin(X_train[:,f])) / (np.amax(X_train[:,f]) - np.amin(X_train[:,f]))   # Normalization
                    X_train[:,f] = (X_train[:,f] - np.amin(X_train[:,f])) / (np.amax(X_train[:,f]) - np.amin(X_train[:,f])) # Normalization
                z_test = (z_test - np.mean(z_train))/np.std(z_train)   # Standardization
                z_train = (z_train - np.mean(z_train))/np.std(z_train) # Standardization

                # OLS with SVD
                b = ridge_solver(X_train, z_train, lmd).flatten()
                
                z_tilde_train[:,j] = X_train @ b
                z_tilde_test[:,j] = X_test @ b

                z_train = z_train.flatten()
                z_test = z_test.flatten()

                mse_train.append(np.mean((z_train - z_tilde_train[:,j])**2))
                mse_test.append(np.mean((z_test - z_tilde_test[:,j])**2))

                r2_train.append(1 - np.sum((z_train - z_tilde_train[:,j])**2)/np.sum((z_train - np.mean(z_train))**2))
                r2_test.append(1 - np.sum((z_test - z_tilde_test[:,j])**2)/np.sum((z_test - np.mean(z_test))**2))

                bia.append(np.mean( (z_test - np.mean(z_tilde_test[:,j]))**2 ))
                var.append(np.var(z_tilde_test[:,j]))

            MSE_train[i-startdeg] = np.mean(mse_train)
            MSE_test[i-startdeg] = np.mean(mse_test)
            R2_train[i-startdeg] = np.mean(r2_train)
            R2_test[i-startdeg] = np.mean(r2_test) 
            bias[i-startdeg] = np.mean(bia)        
            vari[i-startdeg] = np.mean(var)
            
        else:
            # Split into train and test data
            X_train, X_test, z_train, z_test = train_test_split(X, z_, test_size=0.2)

            # scaling 
            for j in range(len(X_test[0,:])):
                X_test[:,j] = (X_test[:,j] - np.amin(X_train[:,j])) / (np.amax(X_train[:,j]) - np.amin(X_train[:,j]))   # Normalization
                X_train[:,j] = (X_train[:,j] - np.amin(X_train[:,j])) / (np.amax(X_train[:,j]) - np.amin(X_train[:,j])) # Normalization
            z_test = (z_test - np.mean(z_train))/np.std(z_train)   # Standardization
            z_train = (z_train - np.mean(z_train))/np.std(z_train) # Standardization
        
            # OLS with SVD
            betas = ridge_solver(X_train, z_train, lmd)
            parameters.append(betas)

            z_tilde_train = np.zeros((len(X_train), 1))
            z_tilde_test = np.zeros((len(X_test), 1))
            
            z_tilde_train = X_train @ betas
            z_tilde_test = X_test @ betas 

            MSE_train[i-startdeg] = np.mean(np.mean((z_train - z_tilde_train)**2, axis=1, keepdims=True))
            MSE_test[i-startdeg] = np.mean(np.mean((z_test - z_tilde_test)**2, axis=1, keepdims=True))
            bias[i-startdeg] = np.mean( (z_test - np.mean(z_tilde_test, axis=1, keepdims=True))**2 )        
            vari[i-startdeg] = np.mean( np.var(z_tilde_test, axis=1, keepdims=True) )
            R2_train[i-startdeg] = 1 - np.sum((z_train - np.mean(z_tilde_train, axis=1, keepdims=True))**2)/np.sum((z_train - np.mean(z_train))**2) # z_?
            R2_test[i-startdeg] = 1 - np.sum((z_test - np.mean(z_tilde_test, axis=1, keepdims=True))**2)/np.sum((z_test - np.mean(z_test))**2) # z_?
    
    # Calculate predicted values using all data (X)
    z_scaled = (z_ - np.mean(z_))/np.std(z_)
    z_tilde = X @ ridge_solver(X, z_scaled, lmd)

    make_plots(x,y,z,z_scaled,z_tilde,startdeg,polydeg,MSE_train,MSE_test,R2_train,R2_test,bias,vari,surface=True)
    # return MSE_test, parameters

def lasso(x, y, z, lmd, polydeg=5, startdeg=1, resampling='None', k=10):
    """
    Description:
        Preforms the LASSO method on the data z. 
        We assume that the data z might depend on x and y.
        The features in the design matrix are polynomials of x and y.
        The LASSO method can be done with bootstraping or cross-validation. 
    Input:
        x (numpy array): Mesh with x values.
        y (numpy array): Mesh with y values.
        z (numpy array): Mesh with z values. This is the data we try to make a fit for.
        polydeg (int): Highest order of polynomial degree in the design matrix.
        resampling (string): 
            Use 'None' for LASSO without resampling.
            Use 'Bootstrap' for LASSO with bootstrapping.
            Use 'CrossValidation' for LASSO with cross-validation.
    Output:
        Plots
    """
    # Format data
    z_ = z.flatten().reshape(-1, 1)

    # Make arrays for MSEs and R2 scores
    MSE_train = np.zeros(polydeg-startdeg+1)
    MSE_test = np.zeros(polydeg-startdeg+1)
    R2_train = np.zeros(polydeg-startdeg+1)
    R2_test = np.zeros(polydeg-startdeg+1)

    # Make lists for parameters and variance
    bias = np.zeros(polydeg-startdeg+1)
    vari = np.zeros(polydeg-startdeg+1)

    parameters = []

    # Loop for OLS with increasing order of polynomial fit
    for i in range(startdeg, polydeg+1):

        # Make design matrix
        X = create_X(x, y, i)
        X = X[:,1:]

        if resampling == "Bootstrap":
            # Split into train and test data
            X_train, X_test, z_train, z_test = train_test_split(X, z_, test_size=0.2)

            # scaling 
            for j in range(len(X_test[0,:])):
                X_test[:,j] = (X_test[:,j] - np.amin(X_train[:,j])) / (np.amax(X_train[:,j]) - np.amin(X_train[:,j]))   # Normalization
                X_train[:,j] = (X_train[:,j] - np.amin(X_train[:,j])) / (np.amax(X_train[:,j]) - np.amin(X_train[:,j])) # Normalization
            z_test = (z_test - np.mean(z_train))/np.std(z_train)   # Standardization
            z_train = (z_train - np.mean(z_train))/np.std(z_train) # Standardization
            
            n_bootstraps = 100

            z_tilde_train = np.zeros((len(X_train), n_bootstraps)) 
            z_tilde_test = np.zeros((len(X_test), n_bootstraps)) 

            for j in range(n_bootstraps): # For each bootstrap

                # Pick new randomly sampled training data and matching design matrix
                indx = np.random.randint(0,len(X_train),len(X_train))
                X_train_btstrp = X_train[indx]
                z_train_btstrp = z_train[indx]

                b = LASSO_solver(X_train_btstrp, z_train_btstrp, lmd)
                z_tilde_train[:,j] = X_train @ b.flatten()
                z_tilde_test[:,j] = X_test @ b.flatten()

            MSE_train[i-startdeg] = np.mean(np.mean((z_train - z_tilde_train)**2, axis=1, keepdims=True))
            MSE_test[i-startdeg] = np.mean(np.mean((z_test - z_tilde_test)**2, axis=1, keepdims=True))
            bias[i-startdeg] = np.mean((z_test - np.mean(z_tilde_test, axis=1, keepdims=True))**2)        
            vari[i-startdeg] = np.mean(np.var(z_tilde_test, axis=1, keepdims=True))
            R2_train[i-startdeg] = 1 - np.sum((z_train - np.mean(z_tilde_train, axis=1, keepdims=True))**2)/np.sum((z_train - np.mean(z_train))**2)
            R2_test[i-startdeg] = 1 - np.sum((z_test - np.mean(z_tilde_test, axis=1, keepdims=True))**2)/np.sum((z_test - np.mean(z_test))**2)

        elif resampling == "CrossValidation":
            # Truncate z_ so to be able to divide into equally sized k folds
            floor = len(z_) // k
            z_cv = z_[:floor*k]
            X_cv = X[:floor*k,:]

            shuffler = np.random.permutation(len(X_cv))
            X_shuffled = X_cv[shuffler]
            z_shuffled = z_cv[shuffler]

            X_groups = np.array_split(X_shuffled, k)
            z_groups = np.array_split(z_shuffled, k)

            z_tilde_train = np.zeros((len(X_cv)-len(X_groups[0]), k))
            z_tilde_test = np.zeros((len(X_groups[0]), k)) 

            mse_train = []
            mse_test = []
            r2_train = []
            r2_test = []
            bia = []
            var = []

            for j in range(k):                
                X_test = X_groups[j] # Picks j'th matrix as test matrix
                X_groups_copy = X_groups # Makes copy of the groups, to not mess up later
                X_groups_reduced = np.delete(X_groups_copy, j, 0) # Remove j'th matrix from the copy-group
                X_train = np.concatenate((X_groups_reduced), axis=0) # Concatenates the remainding matrices

                # Same for the data z 
                z_test = z_groups[j] 
                z_groups_copy = z_groups 
                z_groups_reduced = np.delete(z_groups_copy, j, 0)
                z_train = np.concatenate((z_groups_reduced), axis=0)

                # scaling 
                for f in range(len(X_test[0,:])):
                    X_test[:,f] = (X_test[:,f] - np.amin(X_train[:,f])) / (np.amax(X_train[:,f]) - np.amin(X_train[:,f]))   # Normalization
                    X_train[:,f] = (X_train[:,f] - np.amin(X_train[:,f])) / (np.amax(X_train[:,f]) - np.amin(X_train[:,f])) # Normalization
                z_test = (z_test - np.mean(z_train))/np.std(z_train)   # Standardization
                z_train = (z_train - np.mean(z_train))/np.std(z_train) # Standardization

                # OLS with SVD
                b = LASSO_solver(X_train, z_train, lmd).flatten()
                
                z_tilde_train[:,j] = X_train @ b
                z_tilde_test[:,j] = X_test @ b

                z_train = z_train.flatten()
                z_test = z_test.flatten()

                mse_train.append(np.mean((z_train - z_tilde_train[:,j])**2))
                mse_test.append(np.mean((z_test - z_tilde_test[:,j])**2))

                r2_train.append(1 - np.sum((z_train - z_tilde_train[:,j])**2)/np.sum((z_train - np.mean(z_train))**2))
                r2_test.append(1 - np.sum((z_test - z_tilde_test[:,j])**2)/np.sum((z_test - np.mean(z_test))**2))

                bia.append(np.mean( (z_test - np.mean(z_tilde_test[:,j]))**2 ))
                var.append(np.var(z_tilde_test[:,j]))

            MSE_train[i-startdeg] = np.mean(mse_train)
            MSE_test[i-startdeg] = np.mean(mse_test)
            R2_train[i-startdeg] = np.mean(r2_train)
            R2_test[i-startdeg] = np.mean(r2_test) 
            bias[i-startdeg] = np.mean(bia)        
            vari[i-startdeg] = np.mean(var)
            
        else:
            # Split into train and test data
            X_train, X_test, z_train, z_test = train_test_split(X, z_, test_size=0.2)

            # scaling 
            for j in range(len(X_test[0,:])):
                X_test[:,j] = (X_test[:,j] - np.amin(X_train[:,j])) / (np.amax(X_train[:,j]) - np.amin(X_train[:,j]))   # Normalization
                X_train[:,j] = (X_train[:,j] - np.amin(X_train[:,j])) / (np.amax(X_train[:,j]) - np.amin(X_train[:,j])) # Normalization
            z_test = (z_test - np.mean(z_train))/np.std(z_train)   # Standardization
            z_train = (z_train - np.mean(z_train))/np.std(z_train) # Standardization
        
            # OLS with SVD
            betas = LASSO_solver(X_train, z_train, lmd)
            parameters.append(betas)

            z_tilde_train = np.zeros((len(X_train), 1))
            z_tilde_test = np.zeros((len(X_test), 1))
            
            z_tilde_train = X_train @ betas
            z_tilde_test = X_test @ betas 

            MSE_train[i-startdeg] = np.mean(np.mean((z_train - z_tilde_train)**2, axis=1, keepdims=True))
            MSE_test[i-startdeg] = np.mean(np.mean((z_test - z_tilde_test)**2, axis=1, keepdims=True))
            bias[i-startdeg] = np.mean( (z_test - np.mean(z_tilde_test, axis=1, keepdims=True))**2 )        
            vari[i-startdeg] = np.mean( np.var(z_tilde_test, axis=1, keepdims=True) )
            R2_train[i-startdeg] = 1 - np.sum((z_train - np.mean(z_tilde_train, axis=1, keepdims=True))**2)/np.sum((z_train - np.mean(z_train))**2) # z_?
            R2_test[i-startdeg] = 1 - np.sum((z_test - np.mean(z_tilde_test, axis=1, keepdims=True))**2)/np.sum((z_test - np.mean(z_test))**2) # z_?
    
    # Calculate predicted values using all data (X)
    z_scaled = (z_ - np.mean(z_))/np.std(z_)
    z_tilde = X @ LASSO_solver(X, z_scaled, lmd)

    make_plots(x,y,z,z_scaled,z_tilde,startdeg,polydeg,MSE_train,MSE_test,R2_train,R2_test,bias,vari,surface=True)
    # return MSE_test, parameters

if __name__=="__main__":
    mse_no = np.zeros(7)
    mse_bs = np.zeros(10)
    mse_cv = np.zeros(10)

    # np.random.seed(1998)  # Sets seed so results can be reproduced.

    # Defines domain. No need to scale this data as it's already in the range (0,1)
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y) 

    sigma = .1  # Standard deviation of the noise

    # Franke function with stochastic noise

    """ Each run below runs a regression of the Franke function. """
    # OLS regression


    n = 1
    for i in range(n):
        z = FrankeFunction(x, y) + np.random.normal(0, sigma, x.shape)
        z = (z-np.amin(z))/(np.amax(z)-np.amin(z))
        mse_no += ordinary_least_squares(x, y, z, polydeg=6, startdeg=6, resampling='None')            # No resampling
        # mse_bs += ordinary_least_squares(x, y, z, polydeg=10, startdeg=6, resampling='Bootstrap')       # Bootstrapping
        # mse_cv += ordinary_least_squares(x, y, z, polydeg=10, startdeg=6, resampling='CrossValidation') # 10-fold Cross-Validation

    mse_no /= n
    mse_bs /= n
    mse_cv /= n

    print(mse_no)
    print(mse_bs)
    print(mse_cv)