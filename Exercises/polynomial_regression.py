import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

"""
A number of datapoints y are measured at x. 
We guess that y goes like a second order polynomial.
This code finds the least-squares solution f = Ab to the data y. 
                            f = Ab ≈ y
A is the feature matrix, and b are the parameters that matches the features best.
I say 'best' in the sense that Ab gives the least-squares solution f. 
"""

# Defining data
n = int(1e2)                            # Length of data arrays
x = np.linspace(0,1,n)                  # Array of measuringpoints 
y = 2.0 - 3*x + 5*x**2 + 0.1*np.random.randn(n)    # Array of datapoints 

# Setting up the feature matrix A
poly = PolynomialFeatures(degree=2)                     # Declare the model's features
poly_features = poly.fit_transform(x.reshape(-1, 1))    # Make the feature matrix

# Finding the parameters b and the solution f
poly_reg_model = LinearRegression()         # An instance of the sklearn linear model
poly_reg_model.fit(poly_features, y)        # Finds least-squares solution 
f = poly_reg_model.predict(poly_features)   # Gets array of predicted datapoints

# Plotting
plt.figure(figsize=(10, 6))
plt.title("Polynomial regression)", size=16)
plt.scatter(x, y)
plt.plot(x, f, c="red")
plt.show()

# Error measurements
def R2(y_data, y_model): # Calculates the coefficient of determination
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
print('R2: ', R2(y,f))

def MSE(y_data, y_model): # Calculates the mean square error
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n
print('MSE: ', MSE(y,f))