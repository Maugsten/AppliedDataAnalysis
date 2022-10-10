
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

startdeg = 1
polydeg=10

# mse, b = ordinary_least_squares(x, y, terrain, polydeg, startdeg, resampling='Bootstrap')
# mse = ridge(x, y, terrain, 1e-2, polydeg, startdeg, resampling='Bootstrap')
# mse = lasso(x, y, terrain, 1e-2, polydeg, startdeg, resampling='Bootstrap')


""" The section compares the resampling methods """
# u    = np.zeros(polydeg)
# boot = np.zeros(polydeg)
# k5   = np.zeros(polydeg)
# k10  = np.zeros(polydeg)
# n=100
# for i in range(n):
#     print(int(100*i/n+100/n),'%')
#     ui,b = ordinary_least_squares(x, y, terrain, polydeg, startdeg, resampling='None')
#     u += ui
#     booti, b = ordinary_least_squares(x, y, terrain, polydeg, startdeg, resampling='Bootstrap')
#     boot += booti
#     k5i, b = ordinary_least_squares(x, y, terrain, polydeg, startdeg, resampling='CrossValidation', k=5)
#     k5 += k5i
#     k10i, b = ordinary_least_squares(x, y, terrain, polydeg, startdeg, resampling='CrossValidation', k=10)
#     k10 += k10i 
# u    /= n
# boot /= n
# k5   /= n
# k10  /= n

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
""" End of section """

""" This section makes OLS beta plot """
# mse, beta = ordinary_least_squares(x, y, terrain, polydeg, startdeg, resampling='None')

# plt.figure()
# for i in range(len(beta)):
#     # [print(np.log(beta[i])) for j in range(len(beta[i]))]
#     plt.plot([i+1 for j in range(len(beta[i]))], beta[i], 'o')
# plt.grid()
# plt.title('Parameters for given polynomial degree')
# plt.xlabel('Polynomial degree')
# plt.ylabel('Parameter Coefficient')
# plt.show()
""" End of section """

# ordinary_least_squares(x, y, terrain, polydeg, resampling='None')
# ordinary_least_squares(x, y, terrain, polydeg, resampling='Bootstrap')
# ordinary_least_squares(x, y, terrain, polydeg, resampling='CrossValidation')

""" This section makes a plot for comparison of regression methods for different lambdas """
# lambdas = np.logspace(-3,2,50)
# mse_ols_bootstrap = np.zeros(len(lambdas))
# mse_ols_crossvalidation = np.zeros(len(lambdas))
# mse_ridge_bootstrap = np.zeros(len(lambdas))
# mse_ridge_crossvalidation = np.zeros(len(lambdas))
# mse_lasso_bootstrap = np.zeros(len(lambdas))
# mse_lasso_crossvalidation = np.zeros(len(lambdas))
# for i in range(len(lambdas)):
#     mse_ols_bootstrap[i] = ordinary_least_squares(x, y, terrain, polydeg, startdeg, resampling='Bootstrap')
#     mse_ols_crossvalidation[i] = ordinary_least_squares(x, y, terrain, polydeg, startdeg, resampling='CrossValidation')
#     mse_ridge_bootstrap[i] = ridge(x, y, terrain, lambdas[i], polydeg, startdeg, resampling='Bootstrap')
#     mse_ridge_crossvalidation[i] = ridge(x, y, terrain, lambdas[i], polydeg, startdeg, resampling='CrossValidation')
#     mse_lasso_bootstrap[i] = lasso(x, y, terrain, lambdas[i], polydeg, startdeg, resampling='Bootstrap')
#     mse_lasso_crossvalidation[i] = lasso(x, y, terrain, lambdas[i], polydeg, startdeg, resampling='CrossValidation')

# plt.figure()
# plt.plot(np.log10(lambdas), np.log10(mse_ols_bootstrap), label='OLS (Bootstrap)')
# plt.plot(np.log10(lambdas), np.log10(mse_ols_crossvalidation), label='OLS (10-Fold CV)')
# plt.plot(np.log10(lambdas), np.log10(mse_ridge_bootstrap), label='Ridge (Bootstrap)')
# plt.plot(np.log10(lambdas), np.log10(mse_ridge_crossvalidation), label='Ridge (10-Fold CV)')
# plt.plot(np.log10(lambdas), np.log10(mse_lasso_bootstrap), label='LASSO (Bootstrap)')
# plt.plot(np.log10(lambdas), np.log10(mse_lasso_crossvalidation), label='LASSO (10-Fold CV)')
# plt.title('Comparison of Ridge and LASSO')
# plt.xlabel('log($\lambda$)')
# plt.ylabel('log(MSE)')
# plt.legend()
# plt.show()
""" End of section """

""" This section makes a plot for bias and variance for different lambdas """
# lambdas = np.logspace(-50,2,50)
# mse = np.zeros(len(lambdas))
# bias = np.zeros(len(lambdas))
# vari = np.zeros(len(lambdas))
# for i in range(len(lambdas)):
#     mse[i], bias[i], vari[i] = ridge(x, y, terrain, lambdas[i], polydeg, startdeg, resampling='Bootstrap')

# plt.figure()
# plt.plot(np.log10(lambdas), mse, label='MSE')
# plt.plot(np.log10(lambdas), bias, label='Bias^2')
# plt.plot(np.log10(lambdas), vari, label='Variance')
# plt.title('Hyperparameter dependency of bias and variance (Ridge)')
# plt.xlabel('log($\lambda$)')
# plt.ylabel('MSE')
# plt.legend()
# plt.show()
""" End of section """

""" This section makes heatmaps of Ridge MSE(lmd, polydeg) """
# n = 100
# lambdas = np.logspace(-8,1,10)
# MSE_matrix = np.zeros((len(lambdas), polydeg-startdeg+1))
# for i in range(n):
#     print('Progress: ', 100*i/n, '%')
#     for j in range(len(lambdas)):
#         MSE_matrix[j,:] += ridge(x, y, terrain, lambdas[j], polydeg, startdeg, resampling='None')
# MSE_matrix /= n


# im = plt.imshow(np.log10(MSE_matrix).transpose(), cmap='afmhot', interpolation='nearest')
# ax = plt.gca()
# ax.set_xticks(np.arange(len(lambdas)), labels=np.log10(lambdas))
# ax.set_yticks(range(polydeg-startdeg+1), range(2,polydeg+1))
# # plt.rcParams['text.usetex'] = True
# plt.title('Ridge MSE for different $\lambda$s and polynomial degrees')
# plt.ylabel('Polynomial degree')
# plt.xlabel('log($\lambda$)')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im, cax=cax)
# plt.show()
""" End of section """

""" This section makes heatmaps of LASSO MSE(lmd, polydeg) """
# n = 100
# lambdas = np.logspace(-8,-2,7)
# MSE_matrix = np.zeros((len(lambdas), polydeg-startdeg+1))
# for i in range(n):
#     print('Progress: ', 100*(.01+i/n), '%')
#     for j in range(len(lambdas)):
#         MSE_matrix[j,:] += lasso(x, y, terrain, lambdas[j], polydeg, startdeg, resampling='None')
# MSE_matrix /= n


# im = plt.imshow(np.log10(MSE_matrix).transpose(), cmap='afmhot', interpolation='nearest')
# ax = plt.gca()
# ax.set_xticks(np.arange(len(lambdas)), labels=np.log10(lambdas))
# ax.set_yticks(range(polydeg-startdeg+1), range(2,polydeg+1))
# # plt.rcParams['text.usetex'] = True
# plt.title('LASSO MSE for different $\lambda$s and polynomial degrees')
# plt.ylabel('Polynomial degree')
# plt.xlabel('log($\lambda$)')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im, cax=cax)
# plt.show()
""" End of section """


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
""" End of section """

