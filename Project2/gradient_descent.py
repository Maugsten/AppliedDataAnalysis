
import numpy as np
from sklearn.linear_model import SGDRegressor
from project1_functions import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(2)

def CostFunc(y, X, theta, lmd=0):
    return np.sum((y - X @ theta) @ (y - X @ theta).T) + lmd*(theta @ theta.T)  # should divide expression by 2 (*1/2)

def gradientFunc(X,y,theta,n,lmd=0):    
    return (1.0/n) * X.T @ (X @ (theta) - y) + 2 * lmd * theta

# update the learning rate
t0, t1 = 5, 50
def learning_schedule(t):
    return t0/(t+t1)


def MSE(y, y_predict):
    """ Mean square error

    Args:
        - y
    """
    mse = np.mean(np.mean((y - y_predict)**2, axis=1, keepdims=True))
    return mse

def gradient_descent(X_train, X_test, x_train, y_train, y_test, momentum=0, lmd=0):
    """
    Args:
        X (ndarray):
            n x p design matrix.
        x (array):
            Input variable for y. ???
        y (array):
            The data we want to fit.
    """

    n = len(x_train)

    # ## Analytical ###
    # theta_linreg = np.linalg.inv(X_train.T @ X_train) @ (X_train.T @ y_train)
    # y_pred = np.dot(X_train, theta_linreg)
    # mse = MSE(y_train, y_pred)
    # print(f"MSE analytical train: {mse}")

    # theta_linreg = np.linalg.inv(X_test.T @ X_test) @ (X_test.T @ y_test)
    # y_pred = np.dot(X_test, theta_linreg)
    # mse = MSE(y_test, y_pred)
    # print(f"MSE analytical test: {mse}")

    # ## Scikit-learn ###
    # sgdreg = SGDRegressor(max_iter = 1000, penalty=None, eta0=0.01)
    # sgdreg.fit(X_train, y_train.ravel())
    # y_pred = sgdreg.predict(X_test)
    # mse = mean_squared_error(y_test, y_pred)
    # print(f"\nscikit-learn MSE: {mse}")

    ### Numerical ###

    # initial parameters
    eta = 0.001
    iterations = 1000
    theta_initial = np.random.randn(len(X[0]),1) # initial guess for parameters
    theta = theta_initial
    tol = 10**4  # tolerance for gradient clipping

    # storing the predicted y_train values
    cost_train = [0 for i in range(iterations+1)]
    y_pred_initial = np.dot(X_train, theta)
    cost_train[0] = MSE(y_train, y_pred_initial)

    # storing the predicted y_test values
    cost_test = [0 for i in range(iterations+1)]
    y_pred_initial = np.dot(X_test, theta)
    cost_test[0] = MSE(y_test, y_pred_initial)

    change = 0
    for i in range(iterations):

        # calculating the gradient
        gradient = gradientFunc(X_train,y_train,theta,n,lmd)

        # gradient clipping
        gradient = np.minimum(gradient, tol)

        # update the parameters
        change = eta * gradient + momentum * change
        theta -= change

        # calculating MSE for train and test data
        y_predict = np.dot(X_train, theta)
        cost_train[i+1] = MSE(y_train, y_predict)

        y_predict = np.dot(X_test, theta)
        cost_test[i+1] = MSE(y_test, y_predict)

    x_axis = np.linspace(0,iterations,iterations+1)
    # plt.plot(x_axis, cost_train, label='None')
    plt.plot(x_axis, cost_train, label='None')

    print(f"\nMSE GD train: {cost_train[-1]}")
    print(f"MSE GD test: {cost_test[-1]}\n")

    # AdaGrad
    theta = theta_initial 
    delta = 1e-8  # parameter to avoid possible zero division
    Giter = np.zeros((np.shape(X_train)[1],np.shape(X_train)[1]))  # storing the cumulative gradient

    for i in range(iterations):
        
        # calculate the gradient
        gradient = gradientFunc(X_train,y_train,theta,n,lmd)

        # gradient clipping
        Giter += np.minimum(gradient @ gradient.T, tol)

        # algorithm with only diagonal elements
        Ginverse = np.c_[eta / (delta + np.sqrt(np.diag(Giter)))]
        
        # update the parameters
        change = np.multiply(Ginverse,gradient) + momentum * change
        theta -= change

        # calculating MSE for train and test data
        y_predict = np.dot(X_train, theta)
        cost_train[i+1] = MSE(y_train, y_predict)

        y_predict = np.dot(X_test, theta)
        cost_test[i+1] = MSE(y_test, y_predict)

    # plt.plot(x_axis, cost_train, label='AdaGrad')
    plt.plot(x_axis, cost_train, label='AdaGrad')
    
    print(f"\nMSE GD AdaGrad train: {cost_train[-1]}")
    print(f"MSE GD AdaGrad test: {cost_test[-1]}\n")

    # RMSProp
    theta = theta_initial
    rho = 0.9  # moving average parameter, 0.9 is ususally recommended
    change = 0
    RMS = np.zeros((np.shape(X_train)[1],np.shape(X_train)[1]))

    for i in range(iterations):

        # calculating the gradient
        gradient = gradientFunc(X_train,y_train,theta,n,lmd)
        grad_squared = gradient @ gradient.T 

        # gradient clipping
        grad_squared = np.minimum(grad_squared, tol)

        RMS += rho * RMS + (1 - rho) * grad_squared 

        Ginverse = np.c_[eta / (delta + np.sqrt(np.diag(Giter)))]

        # update the parameters
        change = np.multiply(Ginverse,gradient) + momentum * change
        theta -= change

        # calculating MSE for train and test data
        y_predict = np.dot(X_train, theta)
        cost_train[i+1] = MSE(y_train, y_predict)

        y_predict = np.dot(X_test, theta)
        cost_test[i+1] = MSE(y_test, y_predict)
    # plt.plot(x_axis, cost_train, label='RMSProp')
    plt.plot(x_axis, cost_train, label='RMSProp')
    
    print(f"\nMSE GD RMSProp train: {cost_train[-1]}")
    print(f"MSE GD RMSProp test: {cost_test[-1]}\n")

    # Adam
    theta = theta_initial
    rho1, rho2 = 0.9, 0.999
    m_dw = 0
    v_dw = 0

    for i in range(iterations):
    
        # calculating the gradient
        gradient = gradientFunc(X_train,y_train,theta,n,lmd)
        grad_squared = gradient @ gradient.T

        # gradient clipping
        grad_squared = np.minimum(grad_squared, tol)

        # momentum
        m_dw = rho1 * m_dw + (1 - rho1) * gradient

        # RMS
        v_dw = rho2 * v_dw + (1 - rho2) * grad_squared

        # bias correction
        v_dw_corr = v_dw / (1 - rho2**(i + 1))

        # taking the diagonal only and inverting
        Ginverse_w = np.c_[eta / (delta + np.sqrt(np.diagonal(v_dw_corr)))]
        
        # update the parameters
        change = np.multiply(Ginverse_w,m_dw) + momentum * change
        theta -= change

        # calculating MSE for train and test data
        y_predict = np.dot(X_train, theta)
        cost_train[i+1] = MSE(y_train, y_predict)

        y_predict = np.dot(X_test, theta)
        cost_test[i+1] = MSE(y_test, y_predict)

    # plt.plot(x_axis, cost_train, label='Adam')
    plt.plot(x_axis, cost_train, label='Adam')
    
    print(f"\nMSE GD Adam train: {cost_train[-1]}")
    print(f"MSE GD Adam test: {cost_test[-1]}\n")

    plt.title('Gradient descent with adaptive learning rates.\n Training data', fontsize=20)
    plt.xlabel('iterations', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    # plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.show()


def stochastic_gradient_descent(X_train, X_test, x_train, y_train, y_test, momentum=0, lmd=0):

    theta_initial = np.random.rand(len(X_train[0]),1)  # len(X_train[0]) = number of parameters
    theta = theta_initial
    eta = 0.001  # initial learning rate
    tol = 10**4

    # Stochastic part
    n_epochs = 200  # number of epochs
    M = 5  # batch size
    m = int(len(x_train)/M)  # number of batches

    # create mini batches
    Xbatches = [0 for i in range(m)]
    Ybatches = [0 for i in range(m)]

    for i in range(m):

        random_index = M*np.random.randint(m) 
        Xbatches[i] = X_train[random_index:random_index+M]
        Ybatches[i] = y_train[random_index:random_index+M]

    # storing the predicted y_train values
    cost_train = [0 for i in range(n_epochs + 1)]
    y_pred_initial = np.dot(X_train, theta)
    cost_train[0] = MSE(y_train, y_pred_initial)

    # storing the predicted y_test values
    cost_test = [0 for i in range(n_epochs + 1)]
    y_pred_initial = np.dot(X_test, theta)
    cost_test[0] = MSE(y_test, y_pred_initial)

    change = 0
    for epoch in range(n_epochs):  # looping through epochs
        for i in range(m):  # looping through batches

            # calculate the gradient
            gradient = gradientFunc(Xbatches[i],Ybatches[i],theta,M,lmd)

            # gradient clipping
            gradient = np.minimum(gradient, tol)
            
            # update the parameters
            change = eta * gradient + momentum * change
            theta -= change
        
        # calculating MSE for train and test data
        y_predict = np.dot(X_train, theta)
        cost_train[epoch+1] = MSE(y_train, y_predict)

        y_predict = np.dot(X_test, theta)
        cost_test[epoch+1] = MSE(y_test, y_predict)

    x_axis = np.linspace(0, n_epochs, n_epochs + 1)
    # plt.plot(x_axis, cost_train, label='None')
    plt.plot(x_axis, cost_train, label='None')
    
    print(f"\nMSE SGD None train: {cost_train[-1]}")
    print(f"MSE SGD None test: {cost_test[-1]}\n")


    # AdaGrad
    theta = theta_initial
    delta = 1e-8  # parameter to avoid possible zero division
    change = 0

    for epoch in range(n_epochs):
        Giter = np.zeros((np.shape(X_train)[1],np.shape(X_train)[1]))
        for i in range(m):

            # calculate the gradient
            gradient = gradientFunc(Xbatches[i],Ybatches[i],theta,M,lmd)
            
            # gradient clipping
            Giter += np.minimum(gradient @ gradient.T, tol)

            Ginverse = np.c_[eta / (delta + np.sqrt(np.diag(Giter)))]

            # update the parameters
            change = np.multiply(Ginverse,gradient) + momentum * change
            theta -= change

        # calculating MSE for train and test data
        y_predict = np.dot(X_train, theta)
        cost_train[epoch+1] = MSE(y_train, y_predict)

        y_predict = np.dot(X_test, theta)
        cost_test[epoch+1] = MSE(y_test, y_predict)

    # plt.plot(x_axis, cost_train, label='AdaGrad')
    plt.plot(x_axis, cost_train, label='AdaGrad')
    
    print(f"\nMSE SGD AdaGrad train: {cost_train[-1]}")
    print(f"MSE SGD AdaGrad test: {cost_test[-1]}\n")

    # RMSProp
    theta = theta_initial
    rho = 0.9  # moving average parameter, 0.9 is ususally recommended
    change = 0
    for epoch in range(n_epochs):
        RMS = np.zeros((np.shape(X_train)[1],np.shape(X_train)[1]))
        for i in range(m):

            # calculate the gradient
            gradient = gradientFunc(Xbatches[i],Ybatches[i],theta,M,lmd)
            grad_squared = gradient @ gradient.T 

            # gradient clipping
            grad_squared = np.minimum(grad_squared, tol)

            RMS += rho * RMS + (1 - rho) * grad_squared

            Ginverse = np.c_[eta / (delta + np.sqrt(np.diag(Giter)))]

            # update the parameters
            change = np.multiply(Ginverse,gradient) + momentum * change
            theta -= change

        # calculating MSE for train and test data
        y_predict = np.dot(X_train, theta)
        cost_train[epoch+1] = MSE(y_train, y_predict)

        y_predict = np.dot(X_test, theta)
        cost_test[epoch+1] = MSE(y_test, y_predict)

    # plt.plot(x_axis, cost_train, label='RMSProp')
    plt.plot(x_axis, cost_train, label='RMSProp')
    
    print(f"\nMSE SGD RMSProp train: {cost_train[-1]}")
    print(f"MSE SGD RMSProp test: {cost_test[-1]}\n")


    # Adam
    theta = theta_initial
    rho1, rho2 = 0.9, 0.999

    for epoch in range(n_epochs):
        m_dw = 0
        v_dw = 0
        for i in range(m):
           
            # calculate the gradient
            gradient = gradientFunc(Xbatches[i],Ybatches[i],theta,M,lmd)
            grad_squared = gradient @ gradient.T

            # gradient clipping
            grad_squared = np.minimum(grad_squared, tol)

            # momentum
            m_dw = rho1 * m_dw + (1 - rho1) * gradient

            # RMS
            v_dw = rho2 * v_dw + (1 - rho2) * grad_squared

            # bias correction
            v_dw_corr = v_dw / (1 - rho2**(epoch + 1))

            # taking the diagonal only and inverting
            Ginverse_w = np.c_[eta / (delta + np.sqrt(np.diagonal(v_dw_corr)))]

            # update the parameters
            change = np.multiply(Ginverse_w,m_dw) + momentum * change
            theta -= change

        # calculating MSE for train and test data
        y_predict = np.dot(X_train, theta)
        cost_train[epoch+1] = MSE(y_train, y_predict)

        y_predict = np.dot(X_test, theta)
        cost_test[epoch+1] = MSE(y_test, y_predict)

    # plt.plot(x_axis, cost_train, label='Adam')
    plt.plot(x_axis, cost_train, label='Adam')
    
    print(f"\nMSE SGD Adam train: {cost_train[-1]}")
    print(f"MSE SGD Adam test: {cost_test[-1]}\n")

    plt.title('Stochastic gradient descent with adaptive learning rates\n Training data.', fontsize=16)
    plt.xlabel('epochs', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    # plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.grid()
    plt.show()


if __name__ == "__main__":

    ### check gradient descent ###
    # n = 1000

    # x = 2*np.random.rand(n,1)
    # # y = 4+3*x+np.random.randn(n,1)
    
    # z_ = 1 + 2*x + 3*x**2
    # X = np.c_[np.ones((n,1)), x, x**2]

    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y) 

    sigma = .1  # Standard deviation of the noise
    lmd = .01
    n = 6  # polynomial degree

    # Franke function with stochastic noise
    z = FrankeFunction(x, y) + np.random.normal(0, sigma, x.shape)
    z_ = z.flatten().reshape(-1,1)
    X = create_X(x,y,n)

    x_ = x.flatten().reshape(-1,1)
    X_train, X_test, x_train, x_test, z_train, z_test = train_test_split(X, x_, z_, test_size=0.2)

    momentum = 0.0
    lmd = 0.0
    gradient_descent(X_train, X_test, x_train, z_train, z_test, momentum=momentum, lmd=lmd)
    stochastic_gradient_descent(X_train, X_test, x_train, z_train, z_test, momentum=momentum, lmd=lmd)

    # NOTE: the best eta for GD is 0.01, eta for SGD is 0.001.


