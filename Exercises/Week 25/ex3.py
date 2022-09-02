from cProfile import label
from email import message_from_file
from tracemalloc import start
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed()
n = 100

# making the data set
x = np.linspace(-3, 3, n).reshape(-1,1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1, x.shape)

polydeg = 15
startdeg = 2

MSE_train = np.zeros(polydeg)
MSE_test = np.zeros(polydeg)
R2_train = np.zeros(polydeg)
R2_test = np.zeros(polydeg)

for i in range(startdeg,polydeg):

    # setting up the design matrix with scikit
    poly2 = PolynomialFeatures(degree=i) 
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
        clf = LinearRegression(fit_intercept=False)
        clf.fit(x_data,y_data)
        y_predict = clf.predict(x_data)
        weights = clf.coef_

        return y_predict, weights

    y_predict, weights = OLS(X_train, y_train)
    y_predict_scaled, weights_scaled = OLS(X_train_scaled, y_train_scaled)

    y_predict_test = X_test @ weights.reshape(-1,1)
    y_predict_test_scaled = X_test_scaled @ weights.reshape(-1,1)

    def MSE_R2(y,y_predict):
        mean_sq_err = mean_squared_error(y, y_predict)
        det_coeff = r2_score(y, y_predict)

        return mean_sq_err, det_coeff

    MSE_train[i], R2_train[i] = MSE_R2(y_train, y_predict)
    MSE_train_scaled, R2_train_scaled = MSE_R2(y_train_scaled, y_predict_scaled)
    MSE_test[i], R2_test[i] = MSE_R2(y_test, y_predict_test)
    MSE_test_scaled, R2_test_scaled = MSE_R2(y_test_scaled, y_predict_test_scaled)

MSE_train = MSE_train[startdeg:]
MSE_test = MSE_test[startdeg:]
R2_train = R2_train[startdeg:]
R2_test = R2_test[startdeg:]

x_axis = np.linspace(startdeg,polydeg,polydeg-startdeg)

plt.plot(x_axis, MSE_train, label="Training Sample")
plt.plot(x_axis, MSE_test, label="Test Sample")
plt.title("MSE vs Complexity")
plt.xlabel("Model Complexity")
plt.ylabel("Mean Square Error")
plt.legend()
plt.show()

plt.plot(x_axis, R2_train, label="Training Sample")
plt.plot(x_axis, R2_test, label="Test Sample")
plt.title("R2 vs Complexity")
plt.xlabel("Model Complexity")
plt.ylabel("R2 score")
plt.legend()
plt.show()

"""
Plots before for-loop was made

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

plt.figure("Figure 3")
plt.title("Original data")
plt.scatter(x_test, y_predict_test, label='Fit')
plt.scatter(x_test, y_test, label='Test Data', color='orange', s=15)
plt.legend()
textstr3 = f"MSE = {MSE_test:.3f} \nR2 = {R2_test:.3f}"
plt.text(1.5, 0.2, textstr3, size=12,
         ha="center", va="center",
         bbox=dict(boxstyle="round",
                   fc = "lemonchiffon",
                   alpha = 0.5,
                   )
         )
plt.show()

plt.figure("Figure 4")
plt.title("Scaled data")
plt.scatter(x_test_scaled, y_predict_test_scaled, label='Fit')
plt.scatter(x_test_scaled, y_test_scaled, label='Test Data', color='orange', s=15)
plt.legend()
textstr4 = f"MSE = {MSE_test_scaled:.3f} \nR2 = {R2_test_scaled:.3f}"
plt.text(1.5, 0.2, textstr3, size=12,
         ha="center", va="center",
         bbox=dict(boxstyle="round",
                   fc = "lemonchiffon",
                   alpha = 0.5,
                   )
         )
plt.show()

print(f"MSE_train = {MSE_train:.4f} \nMSE_train_scaled = {MSE_train_scaled:.4f} \nMSE_test = {MSE_test:.4f} \nMSE_test_scaled = {MSE_test_scaled:.4f}")
print(f"R2_train = {R2_train:.4f} \nR2_train_scaled = {R2_train_scaled:.4f} \nR2_test = {R2_test:.4f} \nR2_test_scaled = {R2_test_scaled:.4f}")
"""