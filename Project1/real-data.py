
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from functions import *
from random import random, seed


# load the terrain data
terrain = imread('SRTM_data_Norway_2.tif')
terrain = terrain[::100, ::100] / np.amax(terrain)
np.random.seed(1999)

# # plot the terrain
# plt.figure()
# plt.title('Terrain over Møsvatn Austfjell, Norway')
# plt.imshow(terrain)
# plt.colorbar()
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

x = np.linspace(0,len(terrain[1,:]),len(terrain[1,:]))
y = np.linspace(0,len(terrain[:,1]),len(terrain[:,1]))
x,y = np.meshgrid(x,y)

polydeg=5
# ordinary_least_squares(x,y,terrain,polydeg)
ordinary_least_squares(x, y, terrain, polydeg, resampling='CrossValidation')

