
import numpy as np
from random import random, seed
from functions import *


if __name__ == "__main__":
    np.random.seed(1998)  # Set seed so results can be reproduced.

    # Define domain. No need to scale this data as it's already in the range (0,1)
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)

    sigma = .1  # Standard deviation of the noise
    
    # Franke function with stochastic noise
    z = FrankeFunction(x, y) + np.random.normal(0, sigma, x.shape)

    # Preform regression
    ordinary_least_squares(x, y, z, polydeg=5, resampling='None')
    # ordinary_least_squares(x, y, z, polydeg=5, resampling='Bootstrap')
    # ordinary_least_squares(x, y, z, polydeg=5, resampling='CrossValidation')

    """
    Terminal>>>python filename.py
    (Generates plots)
    """