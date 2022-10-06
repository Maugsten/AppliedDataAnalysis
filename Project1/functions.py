import numpy as np
import matplotlib.pyplot as plt
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

    sigma = .1
    cov_matrix = sigma**2 * np.linalg.pinv(Vt.transpose() @ np.diag(Sigma) @ np.diag(Sigma) @ Vt) 
    betas_variance = np.diag(cov_matrix)

    return betas, betas_variance

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

    """ ----------------------- FIKS DETTE! ----------------------- """
    betas_variance = 0

    return betas, betas_variance

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
    print(betas)


    """ ----------------------- FIKS DETTE! ----------------------- """
    betas_variance = 0

    return betas, betas_variance


def make_plots(x, y, z, z_, z_tilde, startdeg, polydeg, MSE_train, MSE_test, R2_train, R2_test, bias, vari, surface=False):

    # get the correct shape to plot 3D figures
    m = np.shape(z)[0]
    n = np.shape(z)[1]

    z_plot = z_.reshape((m,n))
    z_tilde_plot = z_tilde.reshape((m,n))

    min_val = np.min(z_)
    max_val = np.max(z_)

    # heatmaps
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True) 
    im1 = ax1.imshow(z_plot, vmin=min_val, vmax=max_val)
    ax1.set_title("Original data")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    fig.colorbar(im1, ax=ax1, location='bottom')

    im2 = ax2.imshow(z_tilde_plot, vmin=min_val, vmax=max_val)
    ax2.set_title("OLS fit")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    fig.colorbar(im2, ax=ax2, location='bottom')
    
    im3 = ax3.imshow(z_tilde_plot - z_plot, vmin=min_val, vmax=max_val)
    ax3.set_title("Difference")
    fig.colorbar(im3, ax=ax3, location='bottom')

    if surface == True:
        fig = plt.figure(figsize=plt.figaspect(0.5), constrained_layout=True)
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')

        surf = ax1.plot_surface(x, y, z_plot, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)
        surf1 = ax2.plot_surface(x, y, z_tilde_plot, cmap=cm.coolwarm,
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

    # Plot MSE
    x_axis = np.linspace(startdeg, polydeg, polydeg-startdeg+1)
    plt.figure(figsize=(6, 4))
    plt.plot(x_axis, MSE_train, label="Training Sample")
    plt.plot(x_axis, MSE_test, label="Test Sample")
    plt.title("MSE vs Complexity")
    plt.xlabel("Model Complexity")
    plt.ylabel("Mean Square Error")
    plt.legend()

    # Plot the R2 scores
    plt.figure(figsize=(6, 4))
    plt.plot(x_axis, R2_train, label="Training Sample")
    plt.plot(x_axis, R2_test, label="Test Sample")
    plt.title("R2 score vs Complexity")
    plt.xlabel("Model Complexity")
    plt.ylabel("R2 score")
    plt.legend()

    # Plot the Bias-Variance of test data
    plt.figure(figsize=(6, 4))
    plt.plot(x_axis, MSE_test, label="MSE")
    plt.plot(x_axis, bias, '--', label="Bias")
    plt.plot(x_axis, vari, '--', label="Variance")
    plt.plot(x_axis, bias+vari, '--', label="sum")
    plt.title("Bias-Variance trade off")
    plt.xlabel("Model Complexity")
    plt.ylabel("Error")
    plt.legend()

    plt.show()

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
    # z_ = (z_ - np.mean(z_))/np.std(z_) """ Uncomment for standarization """
    z_ = (z_ - np.min(z_))/(np.max(z_) - np.min(z_)) #""" Uncomment for normalization """

    # Set the first order of polynomials for design matrix to be looped from
    startdeg = 1

    # Make arrays for MSEs and R2 scores
    MSE_train = np.zeros(polydeg-startdeg+1)
    MSE_test = np.zeros(polydeg-startdeg+1)
    R2_train = np.zeros(polydeg-startdeg+1)
    R2_test = np.zeros(polydeg-startdeg+1)

    # Make lists for parameters and variance
    bias = np.zeros(polydeg-startdeg+1)
    vari = np.zeros(polydeg-startdeg+1)

    sklearn_cv_mse = []

    # Loop for OLS with increasing order of polynomial fit
    for i in range(startdeg, polydeg+1):

        # Make design matrix
        X = create_X(x, y, i)

        if resampling == "Bootstrap":
            # Split into train and test data
            X_train, X_test, z_train, z_test = train_test_split(X, z_, test_size=0.2)
            
            n_bootstraps = 100
            beta = np.zeros((len(X_train[0]), n_bootstraps)) 

            z_tilde_train = np.zeros((len(X_train), n_bootstraps)) 
            z_tilde_test = np.zeros((len(X_test), n_bootstraps)) 

            for j in range(n_bootstraps): # For each bootstrap

                # Pick new randomly sampled training data and matching design matrix
                indx = np.random.randint(0,len(X_train),len(X_train))
                X_train_btstrp = X_train[indx]
                z_train_btstrp = z_train[indx]

                # Calculate betas for this bootstrap
                # beta[:,j] = svd_algorithm(X_train_btstrp, z_train_btstrp)[0].flatten()  

                b = svd_algorithm(X_train_btstrp, z_train_btstrp)[0]
                z_tilde_train[:,j] = X_train @ b.flatten()
                z_tilde_test[:,j] = X_test @ b.flatten()

            # Average parameters over the bootstraps
            # betas_averaged = np.mean(beta, axis=1)
            # betas = np.zeros((len(X_train[0]),1))  # prøv reshape her 
            # betas[:,0] = betas_averaged

            # Calculate variances of the parameters
            # betas_variance = np.var(beta, axis=1) 

            MSE_train[i-startdeg] = np.mean(np.mean((z_train - z_tilde_train)**2, axis=1, keepdims=True))
            MSE_test[i-startdeg] = np.mean(np.mean((z_test - z_tilde_test)**2, axis=1, keepdims=True))
            bias[i-startdeg] = np.mean((z_test - np.mean(z_tilde_test, axis=1, keepdims=True))**2)        
            vari[i-startdeg] = np.mean(np.var(z_tilde_test, axis=1, keepdims=True))
            R2_train[i-startdeg] = 1 - np.sum((z_train - np.mean(z_tilde_train, axis=1, keepdims=True))**2)/np.sum((z_train - np.mean(z_))**2)
            R2_test[i-startdeg] = 1 - np.sum((z_test - np.mean(z_tilde_test, axis=1, keepdims=True))**2)/np.sum((z_test - np.mean(z_))**2)
        
        elif resampling == "CrossValidation":
            k = 10

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

                # OLS with SVD
                b = svd_algorithm(X_train, z_train)[0].flatten()
                
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

            # """ To check with sklearn """
            # kf = KFold(n_splits=10, random_state=None)
            # model = LinearRegression() 
            # mse_score = [] 
            # for train_index , test_index in kf.split(X):
            #     X_train , X_test = X[train_index,:],X[test_index,:]
            #     y_train , y_test = z_[train_index] , z_[test_index]
                
            #     model.fit(X_train,y_train)
            #     pred_values = model.predict(X_test)
                
            #     mse = mean_squared_error(pred_values , y_test)
            #     mse_score.append(mse)

            # avg_mse_score = sum(mse_score)/k
            # sklearn_cv_mse.append(avg_mse_score)

            # # print('accuracy of each fold - {}'.format(acc_score))
            # print('Avg accuracy {}: {}'.format(i, avg_mse_score))
            
        else:
            # Split into train and test data
            X_train, X_test, z_train, z_test = train_test_split(X, z_, test_size=0.2)

            # scaler = StandardScaler()
            # scaler.fit(X_train)
            # X_train_scaled = scaler.transform(X_train)
            # X_test_scaled = scaler.transform(X_test)

            # print(np.shape(X_train))
            # print(np.shape(X_train_scaled))
        
            # OLS with SVD
            betas = svd_algorithm(X_train, z_train)[0]

            z_tilde_train = np.zeros((len(X_train), 1))
            z_tilde_test = np.zeros((len(X_test), 1))
            
            z_tilde_train = X_train @ betas
            z_tilde_test = X_test @ betas 

            MSE_train[i-startdeg] = np.mean(np.mean((z_train - z_tilde_train)**2, axis=1, keepdims=True))
            MSE_test[i-startdeg] = np.mean(np.mean((z_test - z_tilde_test)**2, axis=1, keepdims=True))
            bias[i-startdeg] = np.mean( (z_test - np.mean(z_tilde_test, axis=1, keepdims=True))**2 )        
            vari[i-startdeg] = np.mean( np.var(z_tilde_test, axis=1, keepdims=True) )
            R2_train[i-startdeg] = 1 - np.sum((z_train - np.mean(z_tilde_train, axis=1, keepdims=True))**2)/np.sum((z_train - np.mean(z_))**2)
            R2_test[i-startdeg] = 1 - np.sum((z_test - np.mean(z_tilde_test, axis=1, keepdims=True))**2)/np.sum((z_test - np.mean(z_))**2)
    
    # Calculate predicted values using all data (X)
    z_tilde = X @ svd_algorithm(X, z_)[0]

    # plt.figure()
    # plt.plot(range(1,6), sklearn_cv_mse)
    # plt.xlabel('deg')
    # plt.ylabel('mse')
    make_plots(x,y,z,z_,z_tilde,startdeg,polydeg,MSE_train,MSE_test,R2_train,R2_test,bias,vari,surface=True)
    
    

def ridge(x, y, z, lmd, polydeg=5, resampling='None'):
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
    bias = np.zeros(polydeg-startdeg+1)
    vari = np.zeros(polydeg-startdeg+1)

    # Loop for OLS with increasing order of polynomial fit
    for i in range(startdeg, polydeg+1):

        # Make design matrix
        X = create_X(x, y, i)

        if resampling == "Bootstrap":
            # Split into train and test data
            X_train, X_test, z_train, z_test = train_test_split(X, z_, test_size=0.2)

            # Scale data
            z_train_mean = np.mean(z_train)
            z_train_std = np.std(z_train)
            z_train = (z_train - z_train_mean)/z_train_std
            z_test = (z_test - z_train_mean)/z_train_std
            
            n_bootstraps = 100
            beta = np.zeros((len(X_train[0]), n_bootstraps)) 

            z_tilde_train = np.zeros((len(X_train), n_bootstraps)) 
            z_tilde_test = np.zeros((len(X_test), n_bootstraps)) 

            for j in range(n_bootstraps): # For each bootstrap

                # Pick new randomly sampled training data and matching design matrix
                indx = np.random.randint(0,100,len(X_train))
                X_train_btstrp = X_train[indx]
                z_train_btstrp = z_train[indx]

                # Calculate betas for this bootstrap
                beta[:,j] = ridge_solver(X_train_btstrp, z_train_btstrp, lmd)[0].flatten()  

                b = ridge_solver(X_train_btstrp, z_train_btstrp, lmd)[0]
                z_tilde_train[:,j] = X_train @ b.flatten()
                z_tilde_test[:,j] = X_test @ b.flatten()

            # Average parameters over the bootstraps
            betas_averaged = np.mean(beta, axis=1)
            betas = np.zeros((len(X_train[0]),1))  # prøv reshape her 
            betas[:,0] = betas_averaged

            MSE_train[i-startdeg] = np.mean(np.mean((z_train - z_tilde_train)**2, axis=1, keepdims=True))
            MSE_test[i-startdeg] = np.mean(np.mean((z_test - z_tilde_test)**2, axis=1, keepdims=True))
            bias[i-startdeg] = np.mean((z_test - np.mean(z_tilde_test, axis=1, keepdims=True))**2)        
            vari[i-startdeg] = np.mean(np.var(z_tilde_test, axis=1, keepdims=True))
            R2_train[i-startdeg] = 1 - np.sum((z_train - np.mean(z_tilde_train, axis=1, keepdims=True))**2)/np.sum((z_train - np.mean(z_))**2)
            R2_test[i-startdeg] = 1 - np.sum((z_test - np.mean(z_tilde_test, axis=1, keepdims=True))**2)/np.sum((z_test - np.mean(z_))**2)
        
        elif resampling == "CrossValidation":
            k = 10

            # Truncate z_ so to be able to divide into equally sized k folds
            floor = len(z_) // k
            z_cv = z_[:floor*k]
            X_cv = X[:floor*k]

            shuffler = np.random.permutation(len(X_cv))
            X_shuffled = X_cv[shuffler]
            z_shuffled = z_cv[shuffler]

            X_groups = np.array_split(X_shuffled, k)
            z_groups = np.array_split(z_shuffled, k)

            z_tilde_train = np.zeros((len(X_cv)-len(X_groups[0]), k))
            z_tilde_test = np.zeros((len(X_groups[0]), k)) 

            z_tilde_train = np.zeros((len(X_cv)-len(X_groups[0]), k))
            z_tilde_test = np.zeros((len(X_groups[0]), k)) 

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

                # Scale data
                z_train_mean = np.mean(z_train)
                z_train_std = np.std(z_train)
                z_train = (z_train - z_train_mean)/z_train_std
                z_test = (z_test - z_train_mean)/z_train_std

                # OLS with SVD
                b = ridge_solver(X_train, z_train, lmd)[0]
                z_tilde_train = (X_train @ b.flatten()).reshape(-1,1)
                z_tilde_test = (X_test @ b.flatten()).reshape(-1,1)

                """ THIS IS WRONG, BUT ALLOW SCRIPT TO BE RUN """
                betas = ridge_solver(X_train, z_train, lmd)[0] 

            MSE_train[i-startdeg] = np.mean(cv_MSE_train)
            MSE_test[i-startdeg] = np.mean(cv_MSE_test)
            R2_train[i-startdeg] = np.mean(cv_R2_train)
            R2_train[i-startdeg] = np.mean(cv_R2_test)
            bias[i-startdeg] = np.mean(cv_bias)
            vari[i-startdeg] = np.mean(cv_vari)

        else:
            # Split into train and test data
            X_train, X_test, z_train, z_test = train_test_split(X, z_, test_size=0.2)

            # Scale data
            z_train_mean = np.mean(z_train)
            z_train_std = np.std(z_train)
            z_train = (z_train - z_train_mean)/z_train_std
            z_test = (z_test - z_train_mean)/z_train_std
        
            # OLS with SVD
            betas, betas_variance = ridge_solver(X_train, z_train, lmd)

            z_tilde_train = np.zeros((len(X_train), 1))
            z_tilde_test = np.zeros((len(X_test), 1))
            
            z_tilde_train = X_train @ betas
            z_tilde_test = X_test @ betas 

            MSE_train[i-startdeg] = np.mean(np.mean((z_train - z_tilde_train)**2, axis=1, keepdims=True))
            MSE_test[i-startdeg] = np.mean(np.mean((z_test - z_tilde_test)**2, axis=1, keepdims=True))
            bias[i-startdeg] = np.mean((z_test - np.mean(z_tilde_test, axis=1, keepdims=True))**2)        
            vari[i-startdeg] = np.mean(np.var(z_tilde_test, axis=1, keepdims=True))
            R2_train[i-startdeg] = 1 - np.sum((z_train - np.mean(z_tilde_train, axis=1, keepdims=True))**2)/np.sum((z_train - np.mean(z_))**2)
            R2_test[i-startdeg] = 1 - np.sum((z_test - np.mean(z_tilde_test, axis=1, keepdims=True))**2)/np.sum((z_test - np.mean(z_))**2)
        
            MSE_train[i-startdeg] = np.mean(np.mean((z_train - z_tilde_train)**2, axis=1, keepdims=True))
            MSE_test[i-startdeg] = np.mean(np.mean((z_test - z_tilde_test)**2, axis=1, keepdims=True))
            bias[i-startdeg] = np.mean((z_test - np.mean(z_tilde_test, axis=1, keepdims=True))**2)        
            vari[i-startdeg] = np.mean(np.var(z_tilde_test, axis=1, keepdims=True))
            R2_train[i-startdeg] = 1 - np.sum((z_train - np.mean(z_tilde_train, axis=1, keepdims=True))**2)/np.sum((z_train - np.mean(z_))**2)
            R2_test[i-startdeg] = 1 - np.sum((z_test - np.mean(z_tilde_test, axis=1, keepdims=True))**2)/np.sum((z_test - np.mean(z_))**2)
        
    # Calculate predicted values using all data (X)
    z_tilde = X @ ridge_solver(X, z_, lmd)[0]

    make_plots(x,y,z,z_,z_tilde,startdeg,polydeg,MSE_train,MSE_test,R2_train,R2_test,bias,vari,surface=True)


def lasso(x, y, z, lmd, polydeg=5, resampling='None'):
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
    bias = np.zeros(polydeg-startdeg+1)
    vari = np.zeros(polydeg-startdeg+1)

    # Loop for OLS with increasing order of polynomial fit
    for i in range(startdeg, polydeg+1):

        # Make design matrix
        X = create_X(x, y, i)

        if resampling == "Bootstrap":
            # Split into train and test data
            X_train, X_test, z_train, z_test = train_test_split(X, z_, test_size=0.2)

            # # Scale data
            # z_train_mean = np.mean(z_train)
            # z_train_std = np.std(z_train)
            # z_train = (z_train - z_train_mean)/z_train_std
            # z_test = (z_test - z_train_mean)/z_train_std
            
            n_bootstraps = 100
            beta = np.zeros((len(X_train[0]), n_bootstraps)) 

            z_tilde_train = np.zeros((len(X_train), n_bootstraps)) 
            z_tilde_test = np.zeros((len(X_test), n_bootstraps)) 

            for j in range(n_bootstraps): # For each bootstrap

                # Pick new randomly sampled training data and matching design matrix
                indx = np.random.randint(0,100,len(X_train))
                X_train_btstrp = X_train[indx]
                z_train_btstrp = z_train[indx]

                # Calculate betas for this bootstrap
                beta[:,j] = LASSO_solver(X_train_btstrp, z_train_btstrp, lmd)[0].flatten()  

                b = LASSO_solver(X_train_btstrp, z_train_btstrp, lmd)[0]
                z_tilde_train[:,j] = X_train @ b.flatten()
                z_tilde_test[:,j] = X_test @ b.flatten()

            # Average parameters over the bootstraps
            betas_averaged = np.mean(beta, axis=1)
            betas = np.zeros((len(X_train[0]),1))  # prøv reshape her 
            betas[:,0] = betas_averaged

            MSE_train[i-startdeg] = np.mean(np.mean((z_train - z_tilde_train)**2, axis=1, keepdims=True))
            MSE_test[i-startdeg] = np.mean(np.mean((z_test - z_tilde_test)**2, axis=1, keepdims=True))
            bias[i-startdeg] = np.mean((z_test - np.mean(z_tilde_test, axis=1, keepdims=True))**2)        
            vari[i-startdeg] = np.mean(np.var(z_tilde_test, axis=1, keepdims=True))
            R2_train[i-startdeg] = 1 - np.sum((z_train - np.mean(z_tilde_train, axis=1, keepdims=True))**2)/np.sum((z_train - np.mean(z_))**2)
            R2_test[i-startdeg] = 1 - np.sum((z_test - np.mean(z_tilde_test, axis=1, keepdims=True))**2)/np.sum((z_test - np.mean(z_))**2)
        
        elif resampling == "CrossValidation":
            k = 10

            # Truncate z_ so to be able to divide into equally sized k folds
            floor = len(z_) // k
            z_cv = z_[:floor*k]
            X_cv = X[:floor*k]

            shuffler = np.random.permutation(len(X_cv))
            X_shuffled = X_cv[shuffler]
            z_shuffled = z_cv[shuffler]

            X_groups = np.array_split(X_shuffled, k)
            z_groups = np.array_split(z_shuffled, k)

            z_tilde_train = np.zeros((len(X_cv)-len(X_groups[0]), k))
            z_tilde_test = np.zeros((len(X_groups[0]), k)) 

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

                # Scale data
                z_train_mean = np.mean(z_train)
                z_train_std = np.std(z_train)
                z_train = (z_train - z_train_mean)/z_train_std
                z_test = (z_test - z_train_mean)/z_train_std

                # OLS with SVD
                b = LASSO_solver(X_train, z_train, lmd)[0]
                z_tilde_train[:,j] = X_train @ b.flatten()
                z_tilde_test[:,j] = X_test @ b.flatten()

                """ THIS IS WRONG, BUT ALLOW SCRIPT TO BE RUN """
                betas, betas_variance = LASSO_solver(X_train, z_train, lmd) 

            MSE_train[i-startdeg] = np.mean(cv_MSE_train)
            MSE_test[i-startdeg] = np.mean(cv_MSE_test)
            R2_train[i-startdeg] = np.mean(cv_R2_train)
            R2_train[i-startdeg] = np.mean(cv_R2_test)
            bias[i-startdeg] = np.mean(cv_bias)
            vari[i-startdeg] = np.mean(cv_vari)

        else:
            # Split into train and test data
            X_train, X_test, z_train, z_test = train_test_split(X, z_, test_size=0.2)
        
            # # Scale data
            # z_train_mean = np.mean(z_train)
            # z_train_std = np.std(z_train)
            # z_train = (z_train - z_train_mean)/z_train_std
            # z_test = (z_test - z_train_mean)/z_train_std

            # OLS with SVD
            betas, betas_variance = LASSO_solver(X_train, z_train, lmd)

            z_tilde_train = np.zeros((len(X_train), 1))
            z_tilde_test = np.zeros((len(X_test), 1))
            
            z_tilde_train = X_train @ betas
            z_tilde_test = X_test @ betas 

            MSE_train[i-startdeg] = np.mean(np.mean((z_train - z_tilde_train)**2, axis=1, keepdims=True))
            MSE_test[i-startdeg] = np.mean(np.mean((z_test - z_tilde_test)**2, axis=1, keepdims=True))
            bias[i-startdeg] = np.mean((z_test - np.mean(z_tilde_test, axis=1, keepdims=True))**2)        
            vari[i-startdeg] = np.mean(np.var(z_tilde_test, axis=1, keepdims=True))
            R2_train[i-startdeg] = 1 - np.sum((z_train - np.mean(z_tilde_train, axis=1, keepdims=True))**2)/np.sum((z_train - np.mean(z_))**2)
            R2_test[i-startdeg] = 1 - np.sum((z_test - np.mean(z_tilde_test, axis=1, keepdims=True))**2)/np.sum((z_test - np.mean(z_))**2)
        
    # Calculate predicted values using all data (X)
    z_tilde = X @ LASSO_solver(X, z_, lmd)[0]

    make_plots(x,y,z,z_,z_tilde,startdeg,polydeg,MSE_train,MSE_test,R2_train,R2_test,bias,vari,surface=True)
