
from cProfile import label
import numpy as np
from sklearn.linear_model import SGDRegressor
from project1_functions import *
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import grad, elementwise_grad

np.random.seed(2)

def CostFunc(y, X, theta, lmd=0):
    return np.sum((y - X @ theta) @ (y - X @ theta).T) + lmd*(theta @ theta.T)

def gradientFunc(X,y,theta,n,lmd=0):
    return (2.0/n) * X.T @ (X @ (theta) - y) + 2 * lmd * theta

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

def gradient_descent(X, x, y, momentum=0, lmd=0):
    """
    Args:
        X (ndarray):
            n x p design matrix.
        x (array):
            Input variable for y. ???
        y (array):
            The data we want to fit.
    """

    n = len(x)
    XT_X = X.T @ X

    ### Analytical ###
    # I = n*lmd* np.eye(XT_X.shape[0])
    # theta_linreg = np.linalg.inv(XT_X + I) @ X.T @ y
    # print("analytical theta")
    # print(theta_linreg)

    ### Scikit-learn ###
    # sgdreg = SGDRegressor(max_iter = 1000, penalty=None, eta0=0.1)
    # sgdreg.fit(x,y.ravel())
    # print("scikit-learn theta")
    # print(sgdreg.intercept_, sgdreg.coef_)

    ### Numerical ###

    # initial parameters
    eta = 0.01
    iterations = 1000
    theta_initial = np.random.randn(len(X[0]),1) # initial guess for parameters
    theta = theta_initial
    tol = 10**4  # tolerance for gradient clipping

    # storing the predicted y values
    cost = [0 for i in range(iterations+1)]
    y_pred_initial = np.dot(X, theta)
    cost[0] = MSE(y,y_pred_initial)

    change = 0
    for i in range(iterations):

        # calculating the gradient
        gradient = gradientFunc(X,y,theta,n)

        # gradient clipping
        gradient = np.minimum(gradient, tol)

        # update the parameters
        change = eta * gradient + momentum * change
        theta -= change

        # calculating MSE
        y_predict = np.dot(X, theta)
        cost[i+1] = MSE(y, y_predict)

    x_axis = np.linspace(0,iterations,iterations+1)
    plt.plot(x_axis, cost, label='None')
   

    # AdaGrad
    theta = theta_initial 
    delta = 1e-8  # parameter to avoid possible zero division
    Giter = np.zeros((np.shape(X)[1],np.shape(X)[1]))  # storing the cumulative gradient

    for _ in range(iterations):
        
        # calculate the gradient
        gradient = gradientFunc(X,y,theta,n)

        # gradient clipping
        Giter += np.minimum(gradient @ gradient.T, tol)

        # algorithm with only diagonal elements
        Ginverse = np.c_[eta / (delta + np.sqrt(np.diag(Giter)))]
        
        # update the parameters
        change = np.multiply(Ginverse,gradient) + momentum * change
        theta -= change

        # calculating MSE
        y_predict = np.dot(X, theta)
        cost[i+1] = MSE(y, y_predict)

    plt.plot(x_axis, cost, label='AdaGrad')


    # RMSProp
    theta = theta_initial
    rho = 0.9  # moving average parameter, 0.9 is ususally recommended
    change = 0
    RMS = np.zeros((np.shape(X)[1],np.shape(X)[1]))

    for _ in range(iterations):

        # calculating the gradient
        gradient = gradientFunc(X,y,theta,n)
        grad_squared = gradient @ gradient.T 

        # gradient clipping
        grad_squared = np.minimum(grad_squared, tol)

        RMS += rho * RMS + (1 - rho) * grad_squared 

        Ginverse = np.c_[eta / (delta + np.sqrt(np.diag(Giter)))]

        # update the parameters
        change = np.multiply(Ginverse,gradient) + momentum * change
        theta -= change

        # calculating MSE
        y_predict = np.dot(X, theta)
        cost[i+1] = MSE(y, y_predict)

    plt.plot(x_axis, cost, label='RMSProp')

    # Adam
    theta = theta_initial
    rho1, rho2 = 0.9, 0.999
    eta = 0.01
    m_dw = 0
    v_dw = 0

    for i in range(iterations):
    
        # calculating the gradient
        gradient = gradientFunc(X,y,theta,n)
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

        # calculating MSE
        y_predict = np.dot(X, theta)
        cost[i+1] = MSE(y, y_predict)

    plt.plot(x_axis, cost, label='Adam')

    plt.title('Gradient descent with adaptive learning rates')
    plt.xlabel('iterations')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()


def stochastic_gradient_descent(X, x, y, momentum=0, lmd=0):
    
    n = len(x) # number of datapoints

    ## Analytical ###
    theta_linreg = np.linalg.inv(X.T @ X) @ (X.T @ y)

    ### Numerical ###
    theta_initial = np.random.rand(len(X[0]),1)  # len(X[0]) = number of parameters
    theta = theta_initial
    eta = 0.01  # initial learning rate
    tol = 10**4

    # Stochastic part
    n_epochs = 300  # number of epochs
    M = 10  # batch size
    m = int(len(x)/M)  # number of batches

    # create mini batches
    Xbatches = [0 for i in range(m)]
    Ybatches = [0 for i in range(m)]

    for i in range(m):

        random_index = M*np.random.randint(m) 
        Xbatches[i] = X[random_index:random_index+M]
        Ybatches[i] = y[random_index:random_index+M]

    # storing the predicted y values
    cost = [0 for i in range(n_epochs+1)]
    y_pred_initial = np.dot(X, theta)
    cost[0] = MSE(y,y_pred_initial)

    change = 0
    for epoch in range(n_epochs):  # looping through epochs
        for i in range(m):  # looping through batches

            # calculate the gradient
            gradient = gradientFunc(Xbatches[i],Ybatches[i],theta,M)

            # gradient clipping
            gradient = np.minimum(gradient, tol)
            # eta = learning_schedule(epoch * m + i)  # NOTE: seems like updating eta here makes eta too small too fast, MSE gets really bad
            
            # update the parameters
            change = eta * gradient + momentum * change
            theta -= change
        
        # calculating MSE
        y_predict = np.dot(X, theta)
        cost[epoch+1] = MSE(y, y_predict)

    x_axis = np.linspace(0, n_epochs, n_epochs+1)
    plt.plot(x_axis, cost, label='None')


    # AdaGrad
    theta = theta_initial
    delta = 1e-8  # parameter to avoid possible zero division
    change = 0

    for epoch in range(n_epochs):
        Giter = np.zeros((np.shape(X)[1],np.shape(X)[1]))
        for i in range(m):

            # calculate the gradient
            gradient = gradientFunc(Xbatches[i],Ybatches[i],theta,M)
            
            # gradient clipping
            Giter += np.minimum(gradient @ gradient.T, tol)

            Ginverse = np.c_[eta / (delta + np.sqrt(np.diag(Giter)))]

            # update the parameters
            change = np.multiply(Ginverse,gradient) + momentum * change
            theta -= change

        # calculate MSE
        y_predict = np.dot(X, theta)
        cost[epoch+1] = MSE(y, y_predict)

    plt.plot(x_axis, cost, label='AdaGrad')

    # RMSProp
    theta = theta_initial
    rho = 0.9  # moving average parameter, 0.9 is ususally recommended
    change = 0
    for epoch in range(n_epochs):
        RMS = np.zeros((np.shape(X)[1],np.shape(X)[1]))
        for i in range(m):

            # calculate the gradient
            gradient = gradientFunc(Xbatches[i],Ybatches[i],theta,M)
            grad_squared = gradient @ gradient.T 

            # gradient clipping
            grad_squared = np.minimum(grad_squared, tol)

            RMS += rho * RMS + (1 - rho) * grad_squared

            Ginverse = np.c_[eta / (delta + np.sqrt(np.diag(Giter)))]

            # update the parameters
            change = np.multiply(Ginverse,gradient) + momentum * change
            theta -= change

        # calculate MSE
        y_predict = np.dot(X, theta)
        cost[epoch+1] = MSE(y, y_predict)

    plt.plot(x_axis, cost, label='RMSProp')


    # Adam
    theta = theta_initial
    rho1, rho2 = 0.9, 0.999

    for epoch in range(n_epochs):
        m_dw = 0
        v_dw = 0
        for i in range(m):
           
            # calculate the gradient
            gradient = gradientFunc(Xbatches[i],Ybatches[i],theta,M)
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

        # calculate MSE
        y_predict = np.dot(X, theta)
        cost[epoch+1] = MSE(y, y_predict)

    plt.plot(x_axis, cost, label='Adam')

    plt.title('Stochastic gradient descent with adaptive learning rates')
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

if __name__ == "__main__":

    ### check gradient descent ###
    n = 1000

    x = 2*np.random.rand(n,1)
    # y = 4+3*x+np.random.randn(n,1)
    
    z = 1 + 2*x + 3*x**2
    X = np.c_[np.ones((n,1)), x, x**2]

    # x = np.arange(0, 1, 0.05)
    # y = np.arange(0, 1, 0.05)
    # x, y = np.meshgrid(x, y) 

    # sigma = .1  # Standard deviation of the noise
    # lmd = .01
    # n = 6  # polynomial degree

    # # Franke function with stochastic noise
    # z = FrankeFunction(x, y) #+ np.random.normal(0, sigma, x.shape)
    # z = z.flatten().reshape(-1,1)
    # X = create_X(x,y,n)

    gradient_descent(X,x,z)
    # gradient_descent(X,x,y,momentum=0.03)
    # gradient_descent(X,x,z,momentum=0,lmd=1e-3)

    stochastic_gradient_descent(X,x,z)
    # stochastic_gradient_descent(X,x,y,momentum=0.03)
    # stochastic_gradient_descent(X,x,z,momentum=0.03,lmd=1e-3)
