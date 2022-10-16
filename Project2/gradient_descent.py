import numpy as np


def gradient_descent(X, y):
    """
    Args:
        X (ndarray):
            n x p design matrix.
        y (array):
            The data we want to fit.
    """
    
    # number of datapoints
    n = 100

    # Hessian matrix
    H = (2.0/n) * X.T @ X

    # Eigenvalues
    EigValues, EigVectors = np.linalg.eig(H)

    eta = 1.0/np.max(EigValues)
    iterations = 1000
    beta = np.random.randn(2,1)

    for _ in range(iterations):
        gradient = (2.0/n) * X.T @ (X @ beta-y)
        beta -= eta*gradient

def stochastic_gradient_descent():
    pass