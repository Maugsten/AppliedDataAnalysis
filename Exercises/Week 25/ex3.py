import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


np.random.seed()
n = 100
maxdegree = 14

# making the data set
x = np.linspace(-3, 3, n).reshape(-1,1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1, x.shape)

# setting up the design matrix with scikit
poly2 = PolynomialFeatures(degree=maxdegree) 
X = poly2.fit_transform(x)

# split in training and test data
X_train, X_test, y_train, y_test, x_train, x_test = train_test_split(X,y,x,test_size=0.2)

# scaling the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ordinary least squares
clf2 = LinearRegression(fit_intercept=False)
clf2.fit(X_train,y_train)
y_predict = clf2.predict(X_train)

plt.scatter(x_train, y_predict, label='Fit')
plt.scatter(x_train, y_train, label='Data', color='orange', s=15)
plt.legend()
plt.show()