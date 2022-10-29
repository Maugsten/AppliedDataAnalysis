import numpy as np
from sklearn.linear_model import SGDRegressor
import autograd.numpy as np
from autograd import grad, elementwise_grad

np.random.seed(10)

def CostFunc(y, X, theta, lmd=0):
    return np.sum((y - X @ theta)**2) + lmd*theta**2

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
    I = n*lmd* np.eye(XT_X.shape[0])
    theta_linreg = np.linalg.inv(XT_X + I) @ X.T @ y
    print("analytical theta")
    print(theta_linreg)

    ### Scikit-learn ###
    sgdreg = SGDRegressor(max_iter = 50, penalty=None, eta0=0.1)
    sgdreg.fit(x,y.ravel())
    print("scikit-learn theta")
    print(sgdreg.intercept_, sgdreg.coef_)

    ### Numerical ###
    # Hessian matrix
    H = (2.0/n) * XT_X + 2 * lmd * np.eye(XT_X.shape[0])

    # Eigenvalues
    EigValues = np.linalg.eig(H)[0]

    # initial learning rate
    eta = 1.0/np.max(EigValues) # ett forslag til learning rate
    
    # initial parameters
    theta = np.random.randn(len(X[0]),1) # initial guess for parameters

    iterations = 1000
    change = 0
    for _ in range(iterations):
        gradient = (2.0/n) * X.T @ (X @ (theta) - y) + 2 * lmd * theta

        change = eta*gradient + momentum*change
        theta -= change

        # kanskje ta vare på den beste ved hver iterasjon?

    print("theta from GD")
    print(theta)

    # Autograd with AdaGrad
    theta = np.random.randn(len(X[0]),1) # initial guess for parameters
    training_gradient = elementwise_grad(CostFunc,2)
    delta = 1e-8  # AdaGrad parameters to avoid possible zero division
    for _ in range(iterations):
        gradient = training_gradient(y,X,theta)
        # calculate outer product of gradients
        Giter = gradient @ gradient.T
        # algorithm with only diagonal elements
        Ginverse = np.c_[eta/(delta + np.sqrt(np.diag(Giter)))]
        
        change = np.multiply(Ginverse,gradient) + momentum*change
        theta -= change

    print("theta from GD with AdaGrad")
    print(theta)


def stochastic_gradient_descent(X, x, y, momentum=0, lmd=0):
    
    ### Analytical ###
    theta_linreg = np.linalg.inv(X.T @ X) @ (X.T @ y)
    print("analytical theta")
    print(theta_linreg)

    ### Scikit-learn ###
    sgdreg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1)
    sgdreg.fit(x,y.ravel())
    print("scikit-learn theta")
    print(sgdreg.intercept_, sgdreg.coef_)

    ### Numerical ###
    H = (2.0/n)* X.T @ X
    EigValues = np.linalg.eig(H)[0]
    eta = 1.0/np.max(EigValues)
    theta = np.random.rand(len(X[0]),1)  # len(X[0]) = number of parameters

    # Stochastic part
    n_epochs = 50
    M = 5  # batch size
    m = int(n/M)  # number of batches

    t0, t1 = 5, 50
    def learning_schedule(t):
        return t0/(t+t1)

    ### maybe we should divide the batches before the loop? they should be the same for each epoch ###

    change = 0
    for epoch in range(n_epochs):
        for i in range(m):
            random_index = M*np.random.randint(m)
            xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]

            gradients = 2/M * xi.T @ ((xi @ theta) - yi)
            eta = learning_schedule(epoch * m + i)
            
            change = eta*gradients + momentum*change
            theta -= change

    print("theta from SGD")
    print(theta)

    # Autograd with AdaGrad, NOTE: we don't change eta here, why?
    theta = np.random.randn(len(X[0]),1)
    eta = 1.0/np.max(EigValues)
    training_gradient = elementwise_grad(CostFunc,2)  # 2 means we are differentiating with respect to theta
    delta = 1e-8  # AdaGrad parameters to avoid possible zero division
    for epoch in range(n_epochs):
        Giter = np.zeros(shape=(3,3))
        for i in range(m):
            random_index = M*np.random.randint(m)  # why does Morten multiply with M?
            xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]

            gradients = (1.0/M)*training_gradient(yi, xi, theta)
            # calculate outer product of gradients
            Giter += gradients @ gradients.T
            # algorithm with only diagonal elements
            Ginverse = np.c_[eta/(delta + np.sqrt(np.diag(Giter)))]

            change = np.multiply(Ginverse,gradients) + momentum*change
            theta -= change

    print("theta from SGD with AdaGrad")
    print(theta)

    # Autograd with RMSprop, NOTE: we don't change eta here, why?
    theta = np.random.randn(len(X[0]),1)
    # what parameter is this?
    rho = 0.99 
    for epoch in range(n_epochs):
        Giter = np.zeros(shape=(3,3))
        for i in range(m):
            random_index = M*np.random.randint(m)  # why does Morten multiply with M?
            xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]

            gradients = (1.0/M)*training_gradient(yi, xi, theta)
            # previous value for the outer product of gradients
            prev = Giter
            # accumulated gradient
            Giter += gradients @ gradients.T
            # scaling
            Gnew = (rho*prev + (1-rho)*Giter)
            # taking the diagonal and inverting
            Ginverse = np.c_[eta/(delta + np.sqrt(np.diag(Gnew)))]
            
            change = np.multiply(Ginverse,gradients) + momentum*change
            theta -= change

    print("theta from SGD with RMSprop")
    print(theta)

if __name__ == "__main__":

    ### check gradient descent ###
    n = 100

    x = 2*np.random.rand(n,1)
    # y = 4+3*x+np.random.randn(n,1)
    
    y = 3 + 2*x + 3*x**2
    X = np.c_[np.ones((n,1)), x, x**2]

    # gradient_descent(X,x,y)
    # gradient_descent(X,x,y,momentum=0.03)
    gradient_descent(X,x,y,momentum=0.03,lmd=1e-3)
    # stochastic_gradient_descent(X,x,y)
    # stochastic_gradient_descent(X,x,y,momentum=0.03)
    # stochastic_gradient_descent(X,x,y,momentum=0.03,lmd=1e-3)

