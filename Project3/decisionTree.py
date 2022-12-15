import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from mlxtend.evaluate import bias_variance_decomp
# from mlxtend.data import boston_housing_data
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import operator
from sklearn.ensemble import BaggingRegressor

seed = 1
np.random.seed(seed)

# Define dataset
x = np.linspace(0,1,int(2e2))
x0 = .5
s2 = 0.1

y = np.exp(-(x-x0)**2/s2) + np.random.normal(0,.1,len(x))

# Splitting into test and train data
x_train, x_test, y_train, y_test = train_test_split(x.reshape(-1,1), y, test_size=0.2, random_state = seed)
print(x_train.shape[0])
print(x_train.size)

mse = []
bias = []
vari = []

depth = 20

depths = np.linspace(1,depth,depth-1+1)
for i in depths:
    regr = DecisionTreeRegressor(max_depth=i)
    regr.set_params(splitter='random', min_samples_leaf=int(2**i)) #, max_leaf_nodes=int(2**i+1))
    regr.fit(x_train, y_train)
    yHat = regr.predict(x_test)

    tree = DecisionTreeRegressor(max_depth=i)
    tree.set_params(splitter='random', min_samples_leaf=int(2**i)) #, max_leaf_nodes=int(2**i+1))

    # bag = BaggingRegressor(base_estimator=tree,
    #                     n_estimators=10,
    #                     random_state=123)

    # Bias-Variance analysis
    mse_test, bias_current, variance_current = bias_variance_decomp(tree, x_train, y_train, x_test, y_test, loss='mse', num_rounds=10)
    # mse_test, bias_current, variance_current = bias_variance_decomp(bag, x_train, y_train, x_test, y_test, loss='mse', num_rounds=10)

    mse.append(mse_test); bias.append(bias_current); vari.append(variance_current)



plot_tree(regr)
plt.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=depths, y=mse,
                    mode='lines',
                    name='MSE',
                    line=dict(width=2)))
fig.add_trace(go.Scatter(x=depths, y=bias,
                    mode='lines',
                    name='Bias^2',
                    line=dict(width=2)))
fig.add_trace(go.Scatter(x=depths, y=vari,
                    mode='lines',
                    name='Variance',
                    line=dict(width=2)))
fig.update_layout(title='Bias-Variance Tradeoff',
                   xaxis_title='Depth of Tree',
                   yaxis_title='Error')
fig.show()

# print(x_test.T[0])
# print(yHat)

L = sorted(zip(x_test.T[0], yHat), key=operator.itemgetter(0))
new_x_test, new_yHat = zip(*L)

H = sorted(zip(x_train.flatten(), y_train), key=operator.itemgetter(0))
new_x_train, new_y_train = zip(*H)

fig = go.Figure()
fig.add_trace(go.Scatter(x=new_x_train, y=new_y_train,
                    mode='lines',
                    name='Train Data'))
fig.add_trace(go.Scatter(x=x, y=regr.predict(x.reshape(-1,1)),
                    mode='lines',
                    name='Decision Tree'))
fig.add_trace(go.Scatter(x=new_x_test, y=new_yHat,
                    mode='markers',
                    name='Prediction'))
fig.update_layout(title='Decision Tree',
                   xaxis_title='x',
                   yaxis_title='y')
fig.show()