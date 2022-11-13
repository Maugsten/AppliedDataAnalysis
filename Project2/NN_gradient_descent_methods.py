import numpy as np


def gradient_descent(dCdW: list, dCdB: list, W: list, B: list, eta: float, momentum: float, change: list, optimizer: str, RMS_W, RMS_B, M_W, M_B, iteration, dCdW2, dCdB2):
    """
    Args:
  
    """
    # toleranse for gradient clipping
    tol1 = 10**2
    tol2 = 10**4

    if optimizer == "None":
        for i in range(len(W)):
            
            # gradient clipping
            dCdW[i] = np.minimum(dCdW[i], tol1)
            dCdB[i] = np.minimum(dCdB[i], tol1)

            change[i] = eta * dCdW[i] + momentum * change[i]
            W[i] = W[i] - change[i]
            B[i] = B[i] - eta * dCdB[i]


    if optimizer == "AdaGrad": 
        delta = 1e-8  # parameter to avoid possible zero division
        for i in range(len(W)):
        
            # calculate outer product of gradients and perform gradient clipping
            dCdW2[i] += np.minimum(dCdW[i] @ dCdW[i].T, tol2)
            dCdB2[i] += np.minimum(dCdB[i] @ dCdB[i].T, tol2)

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

            # calculate outer product of gradients
            dCdW2_ = dCdW[i] @ dCdW[i].T 
            dCdB2_ = dCdB[i] @ dCdB[i].T

            # gradient clipping
            dCdW2_ = np.minimum(dCdW2_, tol2)
            dCdB2_ = np.minimum(dCdB2_, tol2)
            
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
            
            # gradient clipping
            dCdW2_ = np.minimum(dCdW2_, tol2)
            dCdB2_ = np.minimum(dCdB2_, tol2)

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
