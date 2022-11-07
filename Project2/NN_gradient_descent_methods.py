
import numpy as np


def gradient_descent(dCdW: list, dCdB: list, W: list, B: list, eta: float, momentum: float, change: list, optimizer: str, RMS_W, RMS_B, M_W, M_B, epochNumber):
    """
    Args:
  
    """

    if optimizer == "None":
        for i in range(len(W)):

            change[i] = eta * dCdW[i] + momentum * change[i]
            W[i] = W[i] - change[i]
            B[i] = B[i] - eta * dCdB[i]


    if optimizer == "AdaGrad":
        delta = 1e-8  # AdaGrad parameter to avoid possible zero division
        for i in range(len(W)):
        
            # calculate outer product of gradients
            dCdW2 = dCdW[i] @ dCdW[i].T
            dCdB2 = dCdB[i] @ dCdB[i].T
            # algorithm with only diagonal elements
            Ginverse_W = np.c_[eta / (np.sqrt(dCdW2) + delta)]
            Ginverse_B = np.c_[eta / (np.sqrt(dCdB2) + delta)]
            
            change[i] = np.multiply(Ginverse_W[0,0], dCdW[i]) + momentum * change[i]
            
            W[i] = W[i] - change[i]
            B[i] = B[i] - np.multiply(Ginverse_B[0,0], dCdB[i])


    if optimizer == "RMSprop":
        delta = 1e-8
        rho = 0.9  # moving average parameter, 0.9 is ususally recommended
        for i in range(len(W)):

            dCdW2 = dCdW[i] @ dCdW[i].T 
            dCdB2 = dCdB[i] @ dCdB[i].T
            
            RMS_W[i] = rho * RMS_W[i] + (1 - rho) * dCdW2  
            RMS_B[i] = rho * RMS_B[i] + (1 - rho) * dCdB2

            Ginverse_W = np.c_[eta / (np.sqrt(RMS_W[i]) + delta)]  
            Ginverse_B = np.c_[eta / (np.sqrt(RMS_B[i]) + delta)]
            
            change[i] = np.multiply(Ginverse_W[0,0], dCdW[i]) + momentum * change[i]

            W[i] = W[i] - change[i]
            B[i] = B[i] - np.multiply(Ginverse_B[0,0], dCdB[i])


    if optimizer == "Adam":
        delta = 1e-8
        rho1, rho2 = 0.9, 0.999  # moving average parameter, 0.9 is ususally recommended
        for i in range(len(W)):

            dCdW2 = dCdW[i] @ dCdW[i].T  
            dCdB2 = dCdB[i] @ dCdB[i].T
            
            # RMS
            RMS_W[i] = rho2 * RMS_W[i] + (1 - rho2) * dCdW2  
            RMS_B[i] = rho2 * RMS_B[i] + (1 - rho2) * dCdB2

            # momentum
            M_W[i] = rho1 * M_W[i] + (1 - rho1) * dCdW[i]
            M_B[i] = rho1 * M_B[i] + (1 - rho1) * dCdB[i]

            RMS_W_corr = RMS_W[i] / (1 - np.power(rho2, epochNumber + 1))  
            RMS_B_corr = RMS_B[i] / (1 - np.power(rho2, epochNumber + 1)) 

            M_W_corr = M_W[i] / (1 - np.power(rho1, epochNumber + 1))
            M_B_corr = M_B[i] / (1 - np.power(rho1, epochNumber + 1))

            Ginverse_W = np.c_[eta / (np.sqrt(RMS_W_corr) + delta)]  
            Ginverse_B = np.c_[eta / (np.sqrt(RMS_B_corr) + delta)]
            
            change[i] = np.multiply(Ginverse_W[0,0], M_W_corr) + momentum * change[i]
            W[i] = W[i] - change[i]
            B[i] = B[i] - np.multiply(Ginverse_B[0,0], M_B_corr)

    return W, B


def stochastic_gradient_descent(dCdW: list, dCdB: list, W: list, B: list, eta: float, momentum: float, change: list, optimizer: str, RMS_W, RMS_B, M_W, M_B, epochNumber, batchSize, numOfBatches):
   

    random_index = M*np.random.randint(m)
    xi = X[random_index:random_index+M]
    yi = y[random_index:random_index+M]

    for i in range(m):

        gradients = 2/M * xi.T @ ((xi @ theta) - yi)
        
        change = eta*gradients + momentum*change
        theta -= change

    # Autograd with AdaGrad, NOTE: we don't change eta here, why?
    weights = np.random.randn(len(X[0]),1)
    eta = 1.0/np.max(EigValues)
    training_gradient = elementwise_grad(CostFunc,2)  # 2 means we are differentiating with respect to theta
    
    delta = 1e-8  # AdaGrad parameters to avoid possible zero division
    change = 0
    for epoch in range(n_epochs):
        grad_squared = np.zeros(shape=(3,1))
        for i in range(m):
            random_index = M*np.random.randint(m)  # why does Morten multiply with M?
            xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]

            gradient = (1.0/M)*training_gradient(yi, xi, weights)  # last time we scaled with 2/M, what is correct?
            
            # gradient squared
            grad_squared += gradient * gradient
            # change in weights
            change = (eta/np.sqrt(grad_squared) + delta) * gradient + momentum*change

            weights -= change

    # print("theta from SGD with AdaGrad")
    # print(weights)

    # Autograd with RMSprop, NOTE: we don't change eta here, why?
    weights = np.random.randn(len(X[0]),1)
    # moving average parameter, 0.9 is ususally recommended
    rho = 0.9
    change = 0
    for epoch in range(n_epochs):
        grad_squared = np.zeros(shape=(3,1))
        for i in range(m):
            random_index = M*np.random.randint(m)  # why does Morten multiply with M?
            xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]

            gradient = (1.0/M)*training_gradient(yi, xi, weights)
            prev_grad = grad_squared
            grad_squared = gradient * gradient  # note that this doesn't update prev_grad
            grad_squared += rho*prev_grad + (1-rho)*grad_squared

            old_weights = weights

            change = (eta/np.sqrt(grad_squared) + delta) * gradient + momentum*change
            weights -= change

            # if check_convergence(old_weights, weights):
            #     print(f"converged after {epoch} epochs and {i} iterations")

    # print("theta from SGD with RMSprop")
    # print(weights)

    # Adam
    # picking initially random weights and biases
    theta = np.random.randn(len(X[0]),1)
    #weights = np.random.randn(len(X[0,1:]),1)
    # bias = np.random.randn(1,1)
    #bias = 2.8
    # mean and uncentered variance from the previous time step of the gradients of the parameters
    m_dw, v_dw = 0, 0 
    rho1, rho2 = 0.9, 0.999
    epsilon = 0 #1e-14  # to prevent zero-division

    # these parameters gave the exact solution for the test function
    eta = 0.001
    n_epochs = 300
    M = 5
    m = int(len(x)/M)

    m_dw = 0
    v_dw = 0
    for epoch in range(n_epochs):
        for i in range(m):
            random_index = M*np.random.randint(m)  # why does Morten multiply with M?
            xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]

            # calculating the gradient in terms of weights and biases
            grad_theta = (1.0/M)*training_gradient(yi, xi, theta)
            grad_theta2 = grad_theta @ grad_theta.T

            prev_m_dw = m_dw
            prev_v_dw = v_dw

            ## momentum ##
            m_dw = rho1*prev_m_dw + (1-rho1)*grad_theta

            ## rms ##
            v_dw = rho2*prev_v_dw + (1-rho2)*grad_theta2

            # bias correction
            v_dw_corr = v_dw / (1-rho2**(epoch+1))

            # Taking the diagonal only and inverting
            Ginverse_w = np.c_[eta/(epsilon+np.sqrt(np.diagonal(v_dw_corr)))]
            # Hadamard product
            theta -= np.multiply(Ginverse_w, m_dw)
            
        # print(epoch, theta)
    print("theta from SGD with Adam")
    print(theta)

if __name__ == "__main__":

    ### check gradient descent ###
    n = 1000

    x = 2*np.random.rand(n,1)
    # y = 4+3*x+np.random.randn(n,1)
    
    y = 1 + 2*x + 3*x**2
    X = np.c_[np.ones((n,1)), x, x**2]

    # gradient_descent(X,x,y)
    # gradient_descent(X,x,y,momentum=0.03)
    # gradient_descent(X,x,y,momentum=0.03,lmd=1e-3)
    stochastic_gradient_descent(X,x,y)
    # stochastic_gradient_descent(X,x,y,momentum=0.03)
    # stochastic_gradient_descent(X,x,y,momentum=0.03,lmd=1e-3)

