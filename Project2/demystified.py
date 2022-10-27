"""
Code from Stephen Welch 'Neural Networks Demystified' series. 
https://github.com/stephencwelch/Neural-Networks-Demystified
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import optimize


class Neural_Network(object):
    def __init__(self):        
        #Define Hyperparameters
        self.inputLayerSize = 3
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2
    
    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))
        self.testJ.append(self.N.costFunction(self.testX, self.testY))
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        
        return cost, grad
        
    def train(self, trainX, trainY, testX, testY):
        # Make an internal variable for the callback function:
        self.X = trainX
        self.y = trainY
        
        self.testX = testX
        self.testY = testY

        # Make empty list to store training costs:
        self.J = []
        self.testJ = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(trainX, trainY), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res

def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)
            
            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0
            
        #Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad 

if __name__=="__main__":
    # # Stephen's original data set. InputLayerSize=2
    # X = (hours sleeping, hours studying), y = Score on test
    # X = np.array(([3,5], [5,1], [10,2]), dtype=float)
    # y = np.array(([75], [82], [93]), dtype=float)
 
    n = 100
    x = 2*np.random.rand(n,1)-1
    # y = 3 + 2*x + 3*x**2
    y = np.e**(-x**2)
    X = np.c_[np.ones((n,1)), x, x**2]

    # Scaling the data
    X = X/np.amax(X, axis=0)
    y = y/np.max(y) #Max test score is 100

    # Splitting into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Training and testing our neural network
    NN = Neural_Network()
    T = trainer(NN)
    T.train(X_train, y_train, X_test, y_test)

    plt.figure()
    plt.plot(T.J)
    plt.plot(T.testJ)
    plt.grid(1)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()

    x = np.linspace(-1,1,n)
    # y = 3 + 2*x + 3*x**2
    y = np.e**(-x**2)
    X = np.c_[np.ones((n,1)), x, x**2]

    # Scaling the data
    X = X/np.amax(X, axis=0)
    y = y/np.max(y) #Max test score is 100

    yHat = NN.forward(X)
    plt.figure()
    plt.plot(y)
    plt.plot(yHat, 'o')
    plt.show()

    """
    Note to self: Stephen's suggests adding a regularization 
    parameter. Check if this is a part of the project.
    """

    # numgrad = computeNumericalGradient(NN, X, y)
    # grad = NN.computeGradients(X,y)
    # print(numgrad)
    # print(grad)

    # yHat = NN.forward(X)
    # testValues = np.arange(-5,5,0.01)
    # plt.plot(testValues, NN.sigmoid(testValues), linewidth=2)
    # plt.plot(testValues, NN.sigmoidPrime(testValues), linewidth=2)
    # plt.grid(1)
    # plt.legend(['sigmoid', 'sigmoidPrime'])
    # plt.show()

