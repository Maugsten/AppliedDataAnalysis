import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go

seed = 1

# Define dataset
x = np.linspace(0,1,int(4e2))
x0 = .5
s2 = 0.1

y = np.exp(-(x-x0)**2/s2) + np.random.normal(0,.1,len(x))

# Splitting into test and train data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = seed)
y_train = y_train.ravel()

# Reshaping
input_train = np.c_[x_train]
input_test = np.c_[x_test]

regr = DecisionTreeRegressor(max_depth=5)
regr.fit(input_train, y_train)
yHat = regr.predict(input_test)

# print(input_test)
# print(yHat)

# plt.figure()
# plt.plot(x, y, c="darkorange", label="data")
# plt.scatter(x_test, yHat, color="cornflowerblue", label="max_depth=15")
# # plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Decision Tree Regression")
# plt.legend()
# plt.grid()
# plt.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y,
                    mode='lines',
                    name='lines'))
fig.add_trace(go.Scatter(x=x_test, y=yHat,
                    mode='markers',
                    name='markers'))
fig.show()