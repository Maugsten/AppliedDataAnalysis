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

    # eps = 1  #10**(-8)
    # gradient = (2.0/n) * X.T @ (X @ beta - y)
    # while np.linalg.norm(gradient) > eps:
    #     beta -= eta*gradient
    for _ in range(1000):
        gradient = (2.0/n) * X.T @ (X @ beta - y)
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

def stochastic_gradient_descent(X, x, y, n):
    
    ### Numerical ###
    theta = np.random.rand(2,1)
    eta = 0.1

    for _ in range(1000):
        gradients = 2.0/n * X.T @ ((X @ theta) - y)
        theta -= eta*gradients
    print(theta)

    ### Analytical ###
    theta_linreg = np.linalg.inv(X.T @ X) @ (X.T @ y)
    print(theta_linreg)

    ### Scikit-learn ###
    sgdreg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1)
    sgdreg.fit(x,y.ravel())
    print(sgdreg.intercept_, sgdreg.coef_)

    # Stochastic part
    n_epochs = 50
    t0, t1 = 5, 50
    def learning_schedule(t):
        return t0/(t+t1)

    for epoch in range(n_epochs):
        for i in range(n):
            random_index = np.random.randint(n)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]

            gradients = 2 * xi.T @ ((xi @ theta) - yi)
            eta = learning_schedule(epoch * n + i)
            theta -= eta*gradients
    print(theta)

if __name__ == "__main__":

    ### check gradient descent ###
    n = 100

    x = 2*np.random.rand(n,1)
    y = 4+3*x+np.random.randn(n,1)
    
    X = np.c_[np.ones((n,1)), x]

    stochastic_gradient_descent(X,x,y,n)

