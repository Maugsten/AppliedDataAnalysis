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
def scale(data):
    scaler = StandardScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)

    return scaled_data

X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)
x_train_scaled = scale(x_train)
x_test_scaled = scale(x_test)
y_train_scaled = scale(y_train)
y_test_scaled = scale(y_test)

# ordinary least squares
def OLS(x_data, y_data):
    clf2 = LinearRegression(fit_intercept=False)
    clf2.fit(x_data,y_data)
    y_predict = clf2.predict(x_data)
    weights = clf2.coef_

    return y_predict, weights

y_predict, weights = OLS(X_train, y_train)
y_predict_scaled, weights_scaled = OLS(X_train_scaled, y_train_scaled)

def MSE_R2(y,y_predict):
    mean_sq_err = mean_squared_error(y, y_predict)
    det_coeff = r2_score(y, y_predict)

    return mean_sq_err, det_coeff

MSE_train, R2_train = MSE_R2(y_train, y_predict)
MSE_train_scaled, R2_train_scaled = MSE_R2(y_train_scaled, y_predict_scaled)

plt.figure("Figure 1")
plt.title("Original data")
plt.scatter(x_train, y_predict, label='Fit')
plt.scatter(x_train, y_train, label='Training Data', color='orange', s=15)
plt.legend()
textstr1 = f"MSE = {MSE_train:.3f} \nR2 = {R2_train:.3f}"
plt.text(1.5, 0.2, textstr1, size=12,
         ha="center", va="center",  # horisontal alignment, vertical alignment
         bbox=dict(boxstyle = "round",
                   fc = "lemonchiffon", # facecolor
                   alpha = 0.5,  # transparancy
                   )
         )
plt.show()

plt.figure("Figure 2")
plt.title("Scaled data")
plt.scatter(x_train_scaled, y_predict_scaled, label='Fit')
plt.scatter(x_train_scaled, y_train_scaled, label='Training Data', color='orange', s=15)
plt.legend()
textstr2 = f"MSE = {MSE_train_scaled:.3f} \nR2 = {R2_train_scaled:.3f}"
plt.text(0.75, -1, textstr2, size=12,
         ha="center", va="center",
         bbox=dict(boxstyle="round",
                   fc = "lemonchiffon",
                   alpha = 0.5,
                   )
         )
plt.show()

y_predict_test = X_test @ weights.reshape(-1,1)

plt.figure("Figure 3")
plt.title("Original data")
plt.scatter(x_test, y_predict_test, label='Fit')
plt.scatter(x_test, y_test, label='Test Data', color='orange', s=15)
plt.legend()
textstr2 = f"MSE = {MSE_train_scaled:.3f} \nR2 = {R2_train_scaled:.3f}"
plt.text(0.75, -1, textstr2, size=12,
         ha="center", va="center",
         bbox=dict(boxstyle="round",
                   fc = "lemonchiffon",
                   alpha = 0.5,
                   )
         )
plt.show()
