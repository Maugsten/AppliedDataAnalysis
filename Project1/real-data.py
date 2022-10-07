
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from functions import *
from random import random, seed


# load the terrain data
terrain = imread('SRTM_data_Norway_2.tif')
terrain = terrain[::200, ::200] 
# np.random.seed(98)

# # plot the terrain
# plt.figure()
# plt.title('Terrain over Møsvatn Austfjell, Norway')
# plt.imshow(terrain)
# plt.colorbar()
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

x = np.linspace(0, 1, len(terrain[1,:]))
y = np.linspace(0, 1, len(terrain[:,1]))
x,y = np.meshgrid(x,y)

polydeg=15
# ordinary_least_squares(x, y, terrain, polydeg, resampling='None')
# ordinary_least_squares(x, y, terrain, polydeg, resampling='Bootstrap')
# ordinary_least_squares(x, y, terrain, polydeg, resampling='CrossValidation')

lambdas = np.logspace(-8,2,11)
MSE_matrix = np.zeros((len(lambdas), polydeg))
for i in range(len(lambdas)):
    MSE_matrix[i,:] = ridge(x, y, terrain, lambdas[i], polydeg, resampling='None')

plt.imshow(np.log10(MSE_matrix), cmap='afmhot', interpolation='nearest')
ax = plt.gca()
ax.set_yticks(np.arange(len(lambdas)), labels=lambdas)
plt.title('MSE')
plt.xlabel('Polynomial degree')
plt.ylabel(f'Hyperparameter \lambda')
plt.colorbar()
plt.show()

""" The section compares the resampling methods """
# u=[]
# with open('no_resampling.txt', 'r') as f:
#     u = [[float(num) for num in line.split(',')] for line in f]
# u = np.array(u)[0][:-1]
# print(u)

# boot=[]
# with open('bootstrap.txt', 'r') as f:
#     boot = [[float(num) for num in line.split(',')] for line in f]
# boot = np.array(boot)[0][:-1]
# print(boot)

# k5=[]
# with open('K5.txt', 'r') as f:
#     k5 = [[float(num) for num in line.split(',')] for line in f]
# k5 = np.array(k5)[0][:-1]
# print(k5)

# k10=[]
# with open('K10.txt', 'r') as f:
#     k10 = [[float(num) for num in line.split(',')] for line in f]
# k10 = np.array(k10)[0][:-1]
# print(k10)

# # Plot MSE
# x_axis = range(1,10+1)
# plt.figure(figsize=(6, 4))
# plt.plot(x_axis, u, '--.', label="No Resampling")
# plt.plot(x_axis, boot, '--.', label="Bootstrap")
# plt.plot(x_axis, k5, '--.', label="5-Fold Cross-Validation")
# plt.plot(x_axis, k10, '--.', label="10-Fold Cross-Validation")
# plt.title("MSE vs Complexity")
# plt.xlabel("Polynomial Degree")
# plt.ylabel("Mean Square Error")
# plt.legend()
# plt.show()

