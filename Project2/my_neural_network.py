"""
Code is loosely based on Stephen Welch's 'Neural Networks Demystified'-series. 
https://github.com/stephencwelch/Neural-Networks-Demystified
"""

from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Neural_Network(object):
    def __init__(self, eta=0.1, momentum=0.1):        
        #Define Hyperparameters
        self.inputLayerSize = 3
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)

        #Define learning rate and momentum
        self.eta = eta
        self.momentum = momentum
        
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
    
    def costFunction(self, X, y): # HER KAN DET LEGGES TIL REGULARIZATION
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

    def backpropagate(self):
        dJdW1, dJdW2 = self.costFunctionPrime(self.trainX, self.trainY)

        self.W1 -= self.eta * dJdW1
        self.W2 -= self.eta * dJdW2

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

        max_iterations = 200
        for i in range(max_iterations):
            self.backpropagate()

            self.J.append(self.costFunction(trainX, trainY))
            self.testJ.append(self.costFunction(testX, testY))
            

if __name__=="__main__":
    # Setting the data
    n = 100
    x = 2*np.random.rand(n,1)-1
    noise = np.random.normal(0, .1, (len(x),1))
    y = 3 + 2*x + 3*x**2 + noise
    # y = np.e**(-x**2)

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
    y = 3 + 2*x + 3*x**2 + noise    
    # y = np.e**(-x**2)
    X = np.c_[np.ones((n,1)), x, x**2]
    X = X/np.amax(X, axis=0)
    y = y/np.max(y) 
    yHat = NN.forward(X)
    plt.figure(figsize=(6,4))
    plt.plot(y, label='Real data')
    plt.plot(yHat, 'o', label='Fitted NN data')
    plt.title('Comparison of real data and fitted data')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.legend()
    plt.show()
