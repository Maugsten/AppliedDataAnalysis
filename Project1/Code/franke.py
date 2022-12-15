
import numpy as np
from random import random, seed
from functions import *

np.random.seed(1998)  # Sets seed so results can be reproduced.

# Defines domain. No need to scale this data as it's already in the range (0,1)
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x, y) 

sigma = .1  # Standard deviation of the noise
lmd = .01

# Franke function with stochastic noise
z = FrankeFunction(x, y) + np.random.normal(0, sigma, x.shape)

""" Each run below runs a regression of the Franke function. """
# OLS regression
# ordinary_least_squares(x, y, z, polydeg=10, resampling='None')            # No resampling
ordinary_least_squares(x, y, z, polydeg=10, resampling='Bootstrap')       # Bootstrapping
# ordinary_least_squares(x, y, z, polydeg=10, resampling='CrossValidation') # 10-fold Cross-Validation

# # Ridge regression
# ridge(x, y, z, lmd, polydeg=10, resampling='None')            # No resampling
# ridge(x, y, z, lmd, polydeg=10, resampling='Bootstrap')       # Bootstrapping
# ridge(x, y, z, lmd, polydeg=10, resampling='CrossValidation') # 10-fold Cross-Validation

# # LASSO
# lasso(x, y, z, lmd, polydeg=10, resampling='None')            # No resampling
# lasso(x, y, z, lmd, polydeg=10, resampling='Bootstrap')       # Bootstrapping
# lasso(x, y, z, lmd, polydeg=10, resampling='CrossValidation') # 10-fold Cross-Validation