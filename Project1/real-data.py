
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from functions import *
from random import random, seed


# load the terrain data
terrain = imread('SRTM_data_Norway_2.tif')
terrain = terrain[::20, ::20] 
# np.random.seed(1999)

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

polydeg=7
ordinary_least_squares(x, y, terrain, polydeg)
# ordinary_least_squares(x, y, terrain, polydeg, resampling='CrossValidation')

# l=[]
# with open('data.txt', 'r') as f:
#     l = [[float(num) for num in line.split(',')] for line in f]
# l = np.array(l)[0]
# boot = l[:10]
# k5 = l[10:20]
# k10 = l[20:]

# # Plot MSE
# x_axis = range(1,10+1)
# plt.figure(figsize=(6, 4))
# plt.plot(x_axis, boot, '--.', label="Bootstrap")
# plt.plot(x_axis, k5, '--.', label="5-Fold Cross-Validation")
# plt.plot(x_axis, k10, '--.', label="10-Fold Cross-Validation")
# plt.title("MSE vs Complexity")
# plt.xlabel("Polynomial Degree")
# plt.ylabel("Mean Square Error")
# plt.legend()
# plt.show()

