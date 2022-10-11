
from tkinter import N
from tracemalloc import start
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm 
from functions import *
from random import random, seed


terrain = imread('SRTM_data_Norway_2.tif') # Loads the terrain data
terrain = terrain[::200, ::200] # Downsamples data
np.random.seed(98) # Sets seed for reproducablilty 

x = np.linspace(0, 1, len(terrain[1,:])) # Defines independent variable x
y = np.linspace(0, 1, len(terrain[:,1])) # Defines independent variable y
x,y = np.meshgrid(x,y) # Meshgrid for x and y

startdeg = 1 # Polynomial degree to be loop from
polydeg=10   # Polynomial degree to be loop up to
lmd = 0.01   # Hyperparameter

""" Each run below runs a regression of the Franke function. """
# OLS regression
ordinary_least_squares(x, y, terrain, polydeg=10, resampling='None')            # No resampling
ordinary_least_squares(x, y, terrain, polydeg=10, resampling='Bootstrap')       # Bootstrapping
ordinary_least_squares(x, y, terrain, polydeg=10, resampling='CrossValidation') # 10-fold Cross-Validation

# Ridge regression
ridge(x, y, terrain, lmd, polydeg=10, resampling='None')            # No resampling
ridge(x, y, terrain, lmd, polydeg=10, resampling='Bootstrap')       # Bootstrapping
ridge(x, y, terrain, lmd, polydeg=10, resampling='CrossValidation') # 10-fold Cross-Validation

# LASSO
lasso(x, y, terrain, lmd, polydeg=10, resampling='None')            # No resampling
lasso(x, y, terrain, lmd, polydeg=10, resampling='Bootstrap')       # Bootstrapping
lasso(x, y, terrain, lmd, polydeg=10, resampling='CrossValidation') # 10-fold Cross-Validation