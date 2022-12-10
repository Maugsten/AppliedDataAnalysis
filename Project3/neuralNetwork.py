import numpy as np
from sklearn.neural_network import MLPRegressor
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

# Learning with neural net
regr = MLPRegressor(hidden_layer_sizes=(50, 50, 50),random_state=1, learning_rate_init=0.01, max_iter=50, activation='relu', solver='adam').fit(input_train, y_train)
y_pred = regr.predict(input_test)

# Plotting
# plt.figure(figsize=(6,4))
# plt.plot(x, y, '--', label='y')
# plt.scatter(x_test, y_pred, c='r', label='yHat')
# plt.grid()
# plt.legend()
# plt.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y,
                    mode='lines',
                    name='lines'))
fig.add_trace(go.Scatter(x=x_test, y=y_pred,
                    mode='markers',
                    name='markers'))
fig.show()

# Evaluating
R2 = regr.score(input_test, y_test)
print('R2 Score: ', R2)




# X = np.c_[np.ones(len(x)), x, x**2, x**3]

