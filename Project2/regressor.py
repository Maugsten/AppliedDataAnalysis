"""
Code is loosely based on Stephen Welch's 'Neural Networks Demystified'-series. 
https://github.com/stephencwelch/Neural-Networks-Demystified
"""

from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from NN_gradient_descent_methods import *


class Neural_Network(object):
    def __init__(self, X, numberOfHiddenLayers, nodes, outputLayerSize=1, eta=0.01, lmd=0, momentum=0, epochs=1000, batchSize=10):        
        # Define hyperparameters
        self.X = X # Feature data
        
        self.inputLayerSize = np.shape(self.X)[1] # number of features
        self.outputLayerSize = outputLayerSize    # number of nodes in output layer

        self.numberOfHiddenLayers = numberOfHiddenLayers
        self.nodes = nodes # array with number of nodes in each hidden layer
        
        ## initial Weights and biases for each layer ##
        self.W = []  # will be a nested list where each element is a matrix of weights for one layer
        self.B = []  # will be a nested list where each element is a vector of biases for one layer

        # create weights and biases for each layer
        for i in range(self.numberOfHiddenLayers+2):

            # first layer
            if i == 0:
                # matrix of weights (features x nodes)
                w = np.random.randn(self.inputLayerSize, self.nodes[0])
                self.W.append(w)

                # column vector of biases 
                b = np.random.randn(self.nodes[0])
                self.B.append(b)

            # hidden layers
            elif 0 < i < self.numberOfHiddenLayers:
                # random number of nodes between 2 and 10
                rand_num_nodes = np.random.randint(2,11)

                # matrix of weights (previous number of nodes x current number of nodes)
                w = np.random.randn(self.nodes[i-1], self.nodes[i])
                self.W.append(w)

                # column vector of biases (one bias per node)
                b = np.random.randn(self.nodes[i])
                self.B.append(b)
            
            # output layer
            elif i == self.numberOfHiddenLayers:
                w = np.random.randn(self.nodes[-1], self.outputLayerSize)
                self.W.append(w)
                
                b = np.random.randn(self.outputLayerSize)
                self.B.append(b)

        # More hyperparameters
        self.eta = eta              # Learning rate
        self.momentum = momentum    # Momentum
        self.lmd = lmd              # Regularization parameter
        self.epochs = epochs        # number of iterations through the network
        self.change = [0 for i in range(self.numberOfHiddenLayers+1)]

        # for RMSprop and Adam optimizer method
        self.RMS_W = [0 for i in range(self.numberOfHiddenLayers+1)]
        self.RMS_B = [0 for i in range(self.numberOfHiddenLayers+1)]

        # for Adam optimizer method
        self.M_W = [0 for i in range(self.numberOfHiddenLayers+1)]
        self.M_B = [0 for i in range(self.numberOfHiddenLayers+1)]
        self.epochNumber = [i for i in range(epochs)]

        # for stochastic gradinet descent
        self.batchSize = batchSize
        self.numOfBatches = int(np.shape(self.X)[1] / self.batchSize)
        
        self.z = [0 for i in range(self.numberOfHiddenLayers+1)]
        self.a = [0 for i in range(self.numberOfHiddenLayers+1)]

    def forward(self, X):
        """Propagate inputs through network"""

        for i in range(self.numberOfHiddenLayers+1):

            # first layer
            if i == 0:
                # z[] = np.dot(self.X, W) + B  
                # print(len(self.B))
                self.z[0] = np.dot(X, self.W[0]) + self.B[0]  # NB! numpy turns the vector B into a matrix of the correct shape so that it can be added to the other matrix
                self.a[0] = self.sigmoid(self.z[0])

            # hidden layers
            elif 0 < i <= self.numberOfHiddenLayers:
                self.z[i] = np.dot(self.a[i-1], self.W[i]) + self.B[i]
                self.a[i] = self.sigmoid(self.z[i])         

        yHat = self.a[-1]
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self, z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)

    def ReLU(self, z):
        if z > 0:
            return z
        else: 
            return 0
    
    def ReLUPrime(self, z):
        if z > 0:
            return 1
        else: 
            return 0

    def leakyReLU(self, z, a=0.01):
        if z > 0:
            return z
        else: 
            return a*z

    def leakyReLUPrime(self, z, a=0.01):
        if z > 0:
            return 1
        else: 
            return a
    
    def costFunction(self, X, y): 
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        regularization = sum([np.linalg.norm(self.W[i]) for i in range(len(self.W))])
        
        C = 0.5*sum((y-self.yHat)**2) + self.lmd*regularization
        return C
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and B for a given X and y:
        self.yHat = self.forward(X)

        delta = [0 for i in range(self.numberOfHiddenLayers+1)]
        dCdW = [0 for i in range(self.numberOfHiddenLayers+1)]  # list of derivatives of the cost function in terms of weights
        dCdB = [0 for i in range(self.numberOfHiddenLayers+1)]  # list of derivatives of the cost function in terms of biases

        delta[-1] = np.multiply(-(y-self.a[-1]), self.sigmoidPrime(self.z[-1]))
        dCdW[-1] = np.dot(self.a[-2].T, delta[-1])    
        dCdB[-1] = np.sum(delta[-1], axis=0)

        for i in range(self.numberOfHiddenLayers-1, 0, -1):
            delta[i] = np.dot(delta[i+1], self.W[i+1].T)*self.sigmoidPrime(self.z[i])
            dCdW[i] = np.dot(self.a[i-1].T, delta[i])    
            dCdB[i] = np.sum(delta[i], axis=0) 

        # delta[0] = np.multiply(-(y-self.a[0]), self.sigmoidPrime(self.z[0]))
        delta[0] = np.dot(delta[1], self.W[1].T)*self.sigmoidPrime(self.z[0])
        dCdW[0] = np.dot(self.X.T, delta[0])    
        dCdB[0] = np.sum(delta[0], axis=0) 

        return dCdW, dCdB

    def backpropagate(self):
        dCdW, dCdB = self.costFunctionPrime(self.trainX, self.trainY)

        if self.method == 'GD':
            updatedW, updatedB = gradient_descent(dCdW, dCdB, self.W, self.B, self.eta, self.momentum, self.change, self.optimizer, self.RMS_W, self.RMS_B, self.M_W, self.M_B, self.epoch)
            self.W = updatedW
            self.B = updatedB

        elif self.method == 'SGD':
            updatedW, updatedB = stochastic_gradient_descent(dCdW, dCdB, self.W, self.B, self.eta, self.momentum, self.change, self.RMS_W, self.RMS_B, self.M_W, self.M_B, self.epoch, self.batchSize, self.numOfBatches)
            self.W = updatedW
            self.B = updatedB
        
        else:
            print('Gradient descent method not recognised :(')
            exit()

    def train(self, trainX, trainY, testX, testY, method='GD', optimizer='None'):
        self.method = method
        self.optimizer = optimizer

        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY

        # Make empty list to store training costs:
        self.C = []
        self.testC = []

        self.C.append(self.costFunction(trainX, trainY))
        self.testC.append(self.costFunction(testX, testY))

        # creating minibacthes for stochastic gradient descent
        
        if self.method == 'SGD':
            dCdW_batch = [0 for i in range(self.numOfBatches)]
            dCdB_batch = [0 for i in range(self.numOfBatches)]
            for i in range(self.numOfBatches):
                random_index = self.batchSize*np.random.randint(self.numOfBatches)
                xi = X[random_index:random_index + self.batchSize]
                yi = y[random_index:random_index + self.batchSize]

                dCdW_batch[i] = costFunctionPrime(xi,yi)


        t0, t1 = 5, 50
        def learning_schedule(t):
            return t0/(t+t1)

        for epoch in range(self.epochs):
            self.epoch = epoch
            self.backpropagate() 

            self.C.append(self.costFunction(trainX, trainY))
            self.testC.append(self.costFunction(testX, testY))

            self.eta = learning_schedule(epoch)

    def MSE(self, y):
        mse = np.mean((y-self.yHat)**2)
        return mse
    
    def R2(self, y):
        r2 = 1 - sum((y-self.yHat)**2) / sum((y-np.mean(y))**2) 
        return r2


if __name__=="__main__":
    # np.random.seed(2)
    
    # Setting the data
    n = 1000
    x = np.random.rand(n,1)

    noise = np.random.normal(0, .01, (len(x),1))
    y = 3 + 2*x + 3*x**2 #+ noise
    # y = np.e**(-x**2) #+ noise

    # Scaling the data
    x = x/np.max(x)
    y = y/np.max(y) #Max test score is 100

    # print(np.shape(x))
    # print(np.shape(y))

    # Splitting into train and test data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # nodes = np.array([5, 7, 4])
    nodes = np.array([7, 5])
    NN = Neural_Network(X_train, 2, nodes, outputLayerSize=1, eta=0.001, lmd=0, momentum=0, epochs=1000)
    NN.train(X_train, y_train, X_test, y_test, optimizer='Adam')

    """
    # Training the network
    NN = Neural_Network()
    NN.train(X_train, y_train, X_test, y_test, method='GD')
    """
    # Plotting results
    plt.figure(figsize=(6,4))
    plt.plot(NN.C, label='Train Data')
    plt.plot(NN.testC, label='Test Data')
    plt.grid()
    plt.title('Training our neural network')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()
    
    x = np.linspace(0,1,n).reshape(1,-1).T

    noise = np.random.normal(0, .5, (len(x)))
    y = 3 + 2*x + 3*x**2 #+ noise    
    # y = np.e**(-x**2) #+ noise

    x = x/np.max(x)
    y = y/np.max(y) #Max test score is 100

    yHat = NN.forward(x)
    plt.figure(figsize=(6,4))
    plt.plot(x, y, label='Real data')
    plt.plot(x, yHat, '--', label='Fitted NN data')
    plt.title('Comparison of real data and fitted data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    
