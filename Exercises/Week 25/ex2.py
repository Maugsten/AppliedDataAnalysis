import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

x = np.random.rand(100)
y = 2.0+5*x*x+0.1*np.random.randn(100)

poly2 = PolynomialFeatures(degree=2)  
X = poly2.fit_transform(x[:,np.newaxis])  # creating the design matrix
clf2 = LinearRegression(fit_intercept=False)
clf2.fit(X,y)
Xplot = poly2.fit_transform(x[:,np.newaxis])
y_predict = clf2.predict(Xplot)

plt.plot(np.sort(x), np.sort(y_predict), label='Quadratic Fit')
plt.scatter(x, y, label='Data', color='orange', s=15)
plt.legend()
plt.show()

#Using scikit
MSE = mean_squared_error(y, y_predict)
print(f"Mean square error = {MSE:.4f}")

R2 = r2_score(y, y_predict)
print(f"Coefficient of determination = {R2:.4f}")

#Without scikit
MSE_calc = np.sum((y - y_predict)**2)/len(y)
print(f"Caclulated mean square error = {MSE_calc:.4f}")

y_mean = np.sum(y)/len(y)
R2_calc = 1 - np.sum((y - y_predict)**2)/np.sum((y - y_mean)**2)
print(f"Calculated coefficient of determination = {R2_calc:.4f}")
