
import numpy as np
from random import random, seed
from functions import *


if __name__ == "__main__":
    np.random.seed(1999)  # Set seed so results can be reproduced.

    # Define domain. No need to scale this data as it's already in the range (0,1)
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)

    sigma = .1  # Standard deviation of the noise
    
    # Franke function with stochastic noise
    z = FrankeFunction(x, y) + np.random.normal(0, sigma, x.shape)

    # Preform OLS regression
<<<<<<< HEAD
    # ordinary_least_squares(x, y, z, polydeg=8, resampling='None')
    # ordinary_least_squares(x, y, z, polydeg=8, resampling='Bootstrap')
    ordinary_least_squares(x, y, z, polydeg=8, resampling='CrossValidation')
=======
    # ordinary_least_squares(x, y, z, polydeg=5, resampling='None')
    # ordinary_least_squares(x, y, z, polydeg=6, resampling='Bootstrap')
    ordinary_least_squares(x, y, z, polydeg=5, resampling='CrossValidation')
>>>>>>> 139db0205f5b66a1f058196760e35ffcc6465d1a

    # Preform Ridge regression
    # lmd = .1
    # nlambdas = 9
    # MSERidgePredict = np.zeros(nlambdas)
    # MSELassoPredict = np.zeros(nlambdas)
    # lambdas = np.logspace(-4, 4, nlambdas)
    # for lmd in lambdas:
    #     print('\n Lambda:', lmd)
    #     # ridge(x, y, z, lmd, polydeg=8, resampling='None')
    #     # ridge(x, y, z, lmd, polydeg=8, resampling='Bootstrap')
    #     ridge(x, y, z, lmd, polydeg=8, resampling='CrossValidation')

    # Preform LASSO regression
    # lmd = .1
    # lasso(x, y, z, lmd, polydeg=20, resampling='None')
    # lasso(x, y, z, lmd, polydeg=20, resampling='Bootstrap')
    # lasso(x, y, z, lmd, polydeg=20, resampling='CrossValidation')


    """
    Terminal>>>python filename.py
    (Generates plots)
    """