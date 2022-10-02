import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


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

    """ ----------------------- FIKS DETTE! DU VET IKKE ALLTID SIGMA ----------------------- """
    # cov_matrix = sigma**2 * np.linalg.pinv(Vt.transpose() @ np.diag(Sigma) @ np.diag(Sigma) @ Vt) 
    # betas_variance = np.diag(cov_matrix)
    betas_variance = 0

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
            
            n_bootstraps = 100
            beta = np.zeros((len(X_train[0]), n_bootstraps)) # 1000 is number of bootstraps

            z_predict = np.zeros((len(X_test), n_bootstraps)) # 1000 is number of bootstraps

            for j in range(n_bootstraps): # For each bootstrap
                # Pick new randomly sampled training data and matching design matrix
                indx = np.random.randint(0,100,len(X_train))
                X_train_btstrp = X_train[indx]
                z_train_btstrp = z_train[indx]

                # Calculate betas for this bootstrap
                beta[:,j] = svd_algorithm(X_train_btstrp, z_train_btstrp)[0].flatten()  

                b = svd_algorithm(X_train_btstrp, z_train_btstrp)[0]

                z_predict[:,j] = X_test @ b.flatten()

            # Average parameters over the bootstraps
            betas_averaged = np.mean(beta, axis=1)
            betas = np.zeros((len(X_train[0]),1))  # prøv reshape her 
            betas[:,0] = betas_averaged

            # Calculate variances of the parameters
            betas_variance = np.std(beta, axis=1) 
            """ SJEKK OM DET SKAL VÆRE STD HER """
        
        elif resampling == "CrossValidation":
            k = 10

            shuffler = np.random.permutation(len(X))
            X_shuffled = X[shuffler]
            z_shuffled = z_[shuffler]

            X_groups = np.array_split(X_shuffled, k)
            z_groups = np.array_split(z_shuffled, k)

            z_predict = np.zeros((len(X_groups[0]), k)) 

            for j in range(k):                
                X_test = X_groups[j] # Picks j'th matrix as test matrix
                X_groups_copy = X_groups # Makes copy of the groups, so to not mess up later
                X_groups_reduced = np.delete(X_groups_copy, j, 0) # Remove j'th matrix from the copy-group
                X_train = np.concatenate((X_groups_reduced), axis=0) # Concatenates the remainding matrices

                # Same for the data z 
                z_test = z_groups[j] 
                z_groups_copy = z_groups 
                z_groups_reduced = np.delete(z_groups_copy, j, 0)
                z_train = np.concatenate((z_groups_reduced), axis=0)

                # OLS with SVD
                b = svd_algorithm(X_train, z_train)[0]
                z_predict[:,j] = X_test @ b.flatten()

                """ THIS IS WRONG, BUT ALLOW SCRIPT TO BE RUN """
                betas, betas_variance = svd_algorithm(X_train, z_train) 


        else:
            # Split into train and test data
            X_train, X_test, z_train, z_test = train_test_split(X, z_, test_size=0.2)
        
            # OLS with SVD
            betas, betas_variance = svd_algorithm(X_train, z_train)

        # Calculate training fit and test fit
        z_tilde_train = X_train @ betas
        z_tilde_test = X_test @ betas

        # Collect parameters
        collected_betas.append(betas)
        collected_betas_variance.append(betas_variance)

        # Calculate mean square errors
        MSE_train[i-startdeg] = np.mean((z_train - z_tilde_train)**2)
        # MSE_test[i-startdeg] = np.mean((z_test - z_tilde_test )**2)
        MSE_test[i-startdeg] = np.mean( np.mean((z_test - z_predict)**2, axis=1, keepdims=True) )

        # Calculate mean
        z_mean = np.mean(z_)

        bias[i-startdeg] = np.mean( (z_test - np.mean(z_predict, axis=1, keepdims=True))**2 )
        # bias[i-startdeg] = np.mean( (z_test - np.mean(z_tilde_test))**2 )
        
        vari[i-startdeg] = np.mean( np.var(z_predict, axis=1, keepdims=True) )
        # vari[i-startdeg] = np.mean( np.var(z_tilde_test) )

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

    # Plot of the MSEs
    x_axis = np.linspace(startdeg, polydeg, polydeg-startdeg+1)
    plt.figure(figsize=(6, 4))
    plt.plot(x_axis, np.log(MSE_train), label="Training Sample")
    plt.plot(x_axis, np.log(MSE_test), label="Test Sample")
    plt.title("MSE vs Complexity")
    plt.xlabel("Model Complexity")
    plt.ylabel("Mean Square Error")
    plt.legend()

    # Plot of Bias-Variance of test data
    plt.figure(figsize=(6, 4))
    plt.plot(x_axis, np.log(MSE_test), label="MSE")
    plt.plot(x_axis, np.log(bias), '--', label="Bias")
    plt.plot(x_axis, np.log(vari), '--', label="Variance")
    plt.plot(x_axis, np.log(bias+vari), '--', label="sum")
    plt.title("Bias-Variance trade off")
    plt.xlabel("Model Complexity")
    plt.ylabel("Error")
    plt.legend()

    # Plot of the R2 scores
    plt.figure(figsize=(6, 4))
    plt.plot(x_axis, np.log(R2_train), label="Training Sample")
    plt.plot(x_axis, np.log(R2_test), label="Test Sample")
    plt.title("R2 score vs Complexity")
    plt.xlabel("Model Complexity")
    plt.ylabel("R2 score")
    plt.legend()

    # Plotting paramaters and variance against terms
    plt.figure(figsize=(6, 4))
    for i in range(len(collected_betas)):
        plt.errorbar(range(1, len(collected_betas[i])+1), collected_betas[i].flatten(
        ), collected_betas_variance[i], fmt='o', capsize=6)
    plt.legend(['Order 1', 'Order 2', 'Order 3', 'Order 4', 'Order 5'])
    plt.title('Optimal parameters beta')
    plt.ylabel('Beta value []')
    plt.xlabel('Term index')
    plt.ylim([-200,200])
    plt.show()



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
    collected_betas = []
    collected_betas_variance = []

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
            
            n_bootstraps = 1000
            beta = np.zeros((len(X_train[0]), n_bootstraps)) # 1000 is number of bootstraps

            z_predict = np.zeros((len(X_test), n_bootstraps)) # 1000 is number of bootstraps

            for j in range(n_bootstraps): # For each bootstrap
                # Pick new randomly sampled training data and matching design matrix
                indx = np.random.randint(0,100,len(X_train))
                X_train_btstrp = X_train[indx]
                z_train_btstrp = z_train[indx]

                # Calculate betas for this bootstrap
                beta[:,j] = svd_algorithm(X_train_btstrp, z_train_btstrp)[0].flatten()  

                b = svd_algorithm(X_train_btstrp, z_train_btstrp)[0]

                z_predict[:,j] = X_test @ b.flatten()

            # Average parameters over the bootstraps
            betas_averaged = np.mean(beta, axis=1)
            betas = np.zeros((len(X_train[0]),1))  # prøv reshape her 
            betas[:,0] = betas_averaged

            # Calculate variances of the parameters
            betas_variance = np.std(beta, axis=1)
        
        elif resampling == "CrossValidation":
            k = 10

            shuffler = np.random.permutation(len(X))
            X_shuffled = X[shuffler]
            z_shuffled = z_[shuffler]

            X_groups = np.array_split(X_shuffled, k)
            z_groups = np.array_split(z_shuffled, k)

            z_predict = np.zeros((len(X_groups[0]), k)) 

            for j in range(k):                
                X_test = X_groups[j] # Picks j'th matrix as test matrix
                X_groups_copy = X_groups # Makes copy of the groups, so to not mess up later
                X_groups_reduced = np.delete(X_groups_copy, j, 0) # Remove j'th matrix from the copy-group
                X_train = np.concatenate((X_groups_reduced), axis=0) # Concatenates the remainding matrices

                # Same for the data z 
                z_test = z_groups[j] 
                z_groups_copy = z_groups 
                z_groups_reduced = np.delete(z_groups_copy, j, 0)
                z_train = np.concatenate((z_groups_reduced), axis=0)

                # OLS with SVD
                b = svd_algorithm(X_train, z_train)[0]
                z_predict[:,j] = X_test @ b.flatten()

                """ THIS IS WRONG, BUT ALLOW SCRIPT TO BE RUN """
                betas, betas_variance = svd_algorithm(X_train, z_train) 


        else:
            # Split into train and test data
            X_train, X_test, z_train, z_test = train_test_split(X, z_, test_size=0.2)
        
            # OLS with SVD
            betas, betas_variance = svd_algorithm(X_train, z_train)

        # Calculate training fit and test fit
        z_tilde_train = X_train @ betas
        z_tilde_test = X_test @ betas

        # Collect parameters
        collected_betas.append(betas)
        collected_betas_variance.append(betas_variance)

        # Calculate mean square errors
        MSE_train[i-startdeg] = np.mean((z_train - z_tilde_train)**2)
        MSE_test[i-startdeg] = np.mean((z_test - z_tilde_test )**2)
        # MSE_test[i-startdeg] = np.mean( np.mean((z_test - z_predict)**2, axis=1, keepdims=True) )

        # Calculate mean
        z_mean = np.mean(z_)

        # bias[i-startdeg] = np.mean( (z_test - np.mean(z_predict, axis=1, keepdims=True))**2 )
        
        # vari[i-startdeg] = np.mean( np.var(z_predict, axis=1, keepdims=True) )

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

    # Plot of the MSEs
    x_axis = np.linspace(startdeg, polydeg, polydeg-startdeg+1)
    plt.figure(figsize=(6, 4))
    plt.plot(x_axis, MSE_train, label="Training Sample")
    plt.plot(x_axis, MSE_test, label="Test Sample")
    plt.title("MSE vs Complexity")
    plt.xlabel("Model Complexity")
    plt.ylabel("Mean Square Error")
    plt.legend()

    # Plot of Bias-Variance of test data
    plt.figure(figsize=(6, 4))
    plt.plot(x_axis, MSE_test, label="MSE")
    plt.plot(x_axis, bias, '--', label="Bias")
    plt.plot(x_axis, vari, '--', label="Variance")
    plt.title("Bias-Variance trade off")
    plt.xlabel("Model Complexity")
    plt.ylabel("Error")
    plt.legend()

    # Plot of the R2 scores
    plt.figure(figsize=(6, 4))
    plt.plot(x_axis, R2_train, label="Training Sample")
    plt.plot(x_axis, R2_test, label="Test Sample")
    plt.title("R2 score vs Complexity")
    plt.xlabel("Model Complexity")
    plt.ylabel("R2 score")
    plt.legend()

    # Plotting paramaters and variance against terms
    plt.figure(figsize=(6, 4))
    for i in range(len(collected_betas)):
        plt.errorbar(range(1, len(collected_betas[i])+1), collected_betas[i].flatten(
        ), collected_betas_variance[i], fmt='o', capsize=6)
    plt.legend(['Order 1', 'Order 2', 'Order 3', 'Order 4', 'Order 5'])
    plt.title('Optimal parameters beta')
    plt.ylabel('Beta value []')
    plt.xlabel('Term index')
    plt.ylim([-200,200])
    plt.show()



def lasso(x, y, z, polydeg=5, resampling='None'):
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
    return 0