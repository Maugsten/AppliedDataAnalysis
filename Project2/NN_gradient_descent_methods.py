
import numpy as np


def gradient_descent(dCdW: list, dCdB: list, W: list, B: list, eta: float, momentum: float, change: list, optimizer: str, RMS_W, RMS_B, M_W, M_B, iteration, dCdW2, dCdB2):
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
            dCdW2[i] += dCdW[i] @ dCdW[i].T
            dCdB2[i] += dCdB[i] @ dCdB[i].T
            # algorithm with only diagonal elements
            Ginverse_W = np.c_[eta / (np.sqrt(dCdW2[i]) + delta)]
            Ginverse_B = np.c_[eta / (np.sqrt(dCdB2[i]) + delta)]
            
            change[i] = np.multiply(Ginverse_W[0,0], dCdW[i]) + momentum * change[i]
            
            W[i] = W[i] - change[i]
            B[i] = B[i] - np.multiply(Ginverse_B[0,0], dCdB[i])


    if optimizer == "RMSprop":
        delta = 1e-8
        rho = 0.9  # moving average parameter, 0.9 is ususally recommended
        for i in range(len(W)):

            dCdW2_ = dCdW[i] @ dCdW[i].T 
            dCdB2_ = dCdB[i] @ dCdB[i].T
            
            RMS_W[i] = rho * RMS_W[i] + (1 - rho) * dCdW2_  
            RMS_B[i] = rho * RMS_B[i] + (1 - rho) * dCdB2_

            Ginverse_W = np.c_[eta / (np.sqrt(RMS_W[i]) + delta)]  
            Ginverse_B = np.c_[eta / (np.sqrt(RMS_B[i]) + delta)]
            
            change[i] = np.multiply(Ginverse_W[0,0], dCdW[i]) + momentum * change[i]

            W[i] = W[i] - change[i]
            B[i] = B[i] - np.multiply(Ginverse_B[0,0], dCdB[i])


    if optimizer == "Adam":
        delta = 1e-8
        rho1, rho2 = 0.9, 0.999  # moving average parameter, 0.9 is ususally recommended
        for i in range(len(W)):

            dCdW2_ = dCdW[i] @ dCdW[i].T  
            dCdB2_ = dCdB[i] @ dCdB[i].T
            
            # RMS
            RMS_W[i] = rho2 * RMS_W[i] + (1 - rho2) * dCdW2_  
            RMS_B[i] = rho2 * RMS_B[i] + (1 - rho2) * dCdB2_

            # momentum
            M_W[i] = rho1 * M_W[i] + (1 - rho1) * dCdW[i]
            M_B[i] = rho1 * M_B[i] + (1 - rho1) * dCdB[i]

            RMS_W_corr = RMS_W[i] / (1 - np.power(rho2, iteration + 1))  
            RMS_B_corr = RMS_B[i] / (1 - np.power(rho2, iteration + 1)) 

            M_W_corr = M_W[i] / (1 - np.power(rho1, iteration + 1))
            M_B_corr = M_B[i] / (1 - np.power(rho1, iteration + 1))

            Ginverse_W = np.c_[eta / (np.sqrt(RMS_W_corr) + delta)]  
            Ginverse_B = np.c_[eta / (np.sqrt(RMS_B_corr) + delta)]
            
            change[i] = np.multiply(Ginverse_W[0,0], M_W_corr) + momentum * change[i]
            W[i] = W[i] - change[i]
            B[i] = B[i] - np.multiply(Ginverse_B[0,0], M_B_corr)

    return W, B

"""
def stochastic_gradient_descent(dCdW: list, dCdB: list, W: list, B: list, eta: float, momentum: float, change: list, optimizer: str, RMS_W, RMS_B, M_W, M_B, epoch):
   

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

            RMS_W_corr = RMS_W[i] / (1 - np.power(rho2, epoch + 1))  
            RMS_B_corr = RMS_B[i] / (1 - np.power(rho2, epoch + 1)) 

            M_W_corr = M_W[i] / (1 - np.power(rho1, epoch + 1))
            M_B_corr = M_B[i] / (1 - np.power(rho1, epoch + 1))

            Ginverse_W = np.c_[eta / (np.sqrt(RMS_W_corr) + delta)]  
            Ginverse_B = np.c_[eta / (np.sqrt(RMS_B_corr) + delta)]
            
            change[i] = np.multiply(Ginverse_W[0,0], M_W_corr) + momentum * change[i]
            W[i] = W[i] - change[i]
            B[i] = B[i] - np.multiply(Ginverse_B[0,0], M_B_corr)

    return W, B
"""

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

