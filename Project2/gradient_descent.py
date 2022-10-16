import numpy as np
from sklearn.linear_model import SGDRegressor


def gradient_descent_OLS(X, x, y, n):
    """
    Args:
        X (ndarray):
            n x p design matrix.
        x (array):
            Input variable for y. ???
        y (array):
            The data we want to fit.
    """
    ### Numerical ###
    # Hessian matrix
    H = (2.0/n) * X.T @ X

    # Eigenvalues
    EigValues = np.linalg.eig(H)[0]

    eta = 1.0/np.max(EigValues)
    beta = np.random.randn(2,1)

    eps = 10**(-8)
    gradient = (2.0/n) * X.T @ (X @ beta - y)
    while gradient > eps:
        beta -= eta*gradient

    print(beta)

    ### Analytical ###
    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
    print(beta_linreg)

    ### Scikit-learn ###
    sgdreg = SGDRegressor(max_iter = 50, penalty=None, eta0=0.1)
    sgdreg.fit(x,y.ravel())
    print(sgdreg.intercept_, sgdreg.coef_)

def gradient_descent_ridge(X, x, y, n):

    XT_X = X.T @ X
    lmd = 0.001  # ridge hyperparameter

    ### Numerical ###
    # Hessian matrix
    H = (2.0/n) * XT_X + 2 * lmd * np.eye(XT_X.shape[0])

    # Eigenvalues
    EigValues = np.linalg.eig(H)[0]

    eta = 1.0/np.max(EigValues)
    beta = np.random.randn(2,1)

    eps = 10**(-8)
    gradients = 2.0/n * X.T @ (X @ (beta) - y) + 2 * lmd * beta  # why is it gradient in plural??
    while gradients > eps:
        beta -= eta*gradients

    print(beta)

    ### Analytical ###
    I = n*lmd* np.eye(XT_X.shape[0])
    beta_linreg = np.linalg.inv(XT_X + I) @ X.T @ y
    print(beta_linreg)

def stochastic_gradient_descent():
    pass


n = 100

x = 2*np.random.rand(n,1)
y = 4+3*x+np.random.randn(n,1)

X = np.c_[np.ones((n,1)), x]

gradient_descent_OLS(X,x,y,n)

if __name__ == "main":

    ### check gradient descent ###
    n = 100

    x = 2*np.random.rand(n,1)
    y = 4+3*x+np.random.randn(n,1)
    
    X = np.c_[np.ones((n,1)), x]

    gradient_descent_OLS(X,x,y,n)

