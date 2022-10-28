"""
Code is loosely based on Stephen Welch's 'Neural Networks Demystified'-series. 
https://github.com/stephencwelch/Neural-Networks-Demystified
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Neural_Network(object):
    def __init__(self, eta=0.01, lmd=0, momentum=0, max_iterations=500):        
        # Define hyperparameters
        self.inputLayerSize = 3
        self.outputLayerSize = 1
        self.hiddenLayerSize = 5
        
        # Weights and biases
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        self.B1 = np.zeros(self.hiddenLayerSize) + .01
        self.B2 = np.zeros(self.outputLayerSize) + .01

        # More hyperparameters
        self.eta = eta              # Learning rate
        self.momentum = momentum    # Momentum
        self.lmd = lmd              # Regularization parameter
        self.max_iterations = max_iterations

    def forward(self, X):
        #Propogate inputs though network
        
        self.z2 = np.dot(X, self.W1) + self.B1 
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2) + self.B2
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self, z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)

    def relu(self, z):
        if z>0:
            return z
        else: 
            return 0
    
    def reluPrime(self, z):
        if z>0:
            return 1
        else: 
            return 0

    def leakyReLU(self, z, a=0.01):
        if z>0:
            return z
        else: 
            return a*z

    def leakyReLUPrime(self, z, a=0.01):
        if z>0:
            return 1
        else: 
            return a
    
    def costFunction(self, X, y): # HER KAN DET LEGGES TIL REGULARIZATION
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        C = 0.5*sum((y-self.yHat)**2) + self.lmd*(np.linalg.norm(self.W1) + np.linalg.norm(self.W2))
        return C
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2, np.sum(delta2, axis=0), np.sum(delta3, axis=0)

    def backpropagate(self):
        dJdW1, dJdW2, dJdB1, dJdB2 = self.costFunctionPrime(self.trainX, self.trainY)

        """ HER KAN VI IMPLEMENTERE FLERE GD METODER """
        self.W1 = self.W1 - self.eta * dJdW1
        self.W2 = self.W2 - self.eta * dJdW2
        
        self.B1 = self.B1 - self.eta * dJdB1
        self.B2 = self.B2 - self.eta * dJdB2
        

    def train(self, trainX, trainY, testX, testY):
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY

        # Make empty list to store training costs:
        self.J = []
        self.testJ = []

        self.J.append(self.costFunction(trainX, trainY))
        self.testJ.append(self.costFunction(testX, testY))

        for i in range(self.max_iterations):
            self.backpropagate()

            self.J.append(self.costFunction(trainX, trainY))
            self.testJ.append(self.costFunction(testX, testY))

    def MSE(self, y):
        mse = np.mean((y-self.yHat)**2)
        return mse
    
    def R2(self, y):
        r2 = 1 - sum((y-self.yHat)**2) / sum((y-np.mean(y))**2) 
        return r2


if __name__=="__main__":
    # Setting the data
    n = 1000
    x = 2*np.random.rand(n,1)-1
    noise = np.random.normal(0, .01, (len(x),1))
    # y = 3 + 2*x + 3*x**2 + noise
    y = np.e**(-x**2) #+ noise

    # Making the design matrix
    X = np.c_[np.ones((n,1)), x, x**2]

    # Scaling the data
    X = X/np.amax(X, axis=0)
    y = y/np.max(y) #Max test score is 100

    # Splitting into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Training the network
    NN = Neural_Network()
    NN.train(X_train, y_train, X_test, y_test)

    # Plotting results
    plt.figure(figsize=(6,4))
    plt.plot(NN.J)
    plt.plot(NN.testJ)
    plt.grid()
    plt.title('Training our neural network')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()

    x = np.linspace(-1,1,n)
    noise = np.random.normal(0, .5, (len(x)))
    # y = 3 + 2*x + 3*x**2 + noise    
    y = np.e**(-x**2) #+ noise
    X = np.c_[np.ones((n,1)), x, x**2]
    X = X/np.amax(X, axis=0)
    y = y/np.max(y) 
    yHat = NN.forward(X)
    plt.figure(figsize=(6,4))
    plt.plot(x, y, label='Real data')
    plt.plot(x, yHat, '--', label='Fitted NN data')
    plt.title('Comparison of real data and fitted data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
