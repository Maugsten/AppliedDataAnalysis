import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

x = np.random.rand(100)
y = 2.0+5*x*x+0.1*np.random.randn(100)

poly2 = PolynomialFeatures(degree=2)
X = poly2.fit_transform(x[:,np.newaxis])
clf2 = LinearRegression()
clf2.fit(X,y)
Xplot=poly2.fit_transform(x[:,np.newaxis])

plt.plot(np.sort(x), np.sort(clf2.predict(Xplot)), label='Quadratic Fit')
plt.scatter(x, y, label='Data', color='orange', s=15)
plt.legend()
plt.show()
