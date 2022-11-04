"""
Code is loosely based on Stephen Welch's 'Neural Networks Demystified'-series. 
https://github.com/stephencwelch/Neural-Networks-Demystified
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Neural_Network(object):
    def __init__(self, X, numberOfHiddenLayers=1, nodes=1, outputLayerSize=1, eta=0.01, lmd=0, momentum=0, maxIterations=500):        
        # Define hyperparameters
        # self.inputLayerSize = 1
        # self.hiddenLayerSize = 5
        # self.outputLayerSize = 1

        self.X = X # Feature data
        
        self.inputLayerSize = np.shape(self.X)[1] # number of features
        self.outputLayerSize = outputLayerSize    # number of nodes in output layer

        self.numberOfHiddenLayers = numberOfHiddenLayers
        self.nodes = nodes # array with number of nodes in each hidden layer
        
        ## initial Weights and biases for each layer ##
        # prev_num_nodes = self.nodes

        self.W = []
        self.B = []

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

                # matrix of weights (previous number of nodes x random number of nodes)
                w = np.random.randn(self.nodes[i-1], self.nodes[i])
                self.W.append(w)
                # column vector of biases (one bias per node)
                
                b = np.random.randn(self.nodes[i])
                self.B.append(b)
                # prev_num_nodes = rand_num_nodes 
            
            # output layer
            elif i == self.numberOfHiddenLayers:
                w = np.random.randn(self.nodes[-1], self.outputLayerSize)
                self.W.append(w)
                
                b = np.random.randn(self.outputLayerSize,1)
                self.B.append(b)

        # self.W1 = np.random.randn(self.features,self.nodes)  # features x nodes, first layer. This shape means that we don't have to transpose the matrix
        # self.W2 = np.random.randn(self.features,self.nodes) # second layer
        # self.B1 = np.zeros(self.hiddenLayerSize) + .01  # i.e. 10 zeros
        # self.B2 = np.zeros(self.outputLayerSize) + .01  # i.e. 1 zero


        # More hyperparameters
        self.eta = eta              # Learning rate
        self.momentum = momentum    # Momentum
        self.lmd = lmd              # Regularization parameter
        self.maxIterations = maxIterations
        self.change = [0 for i in range(self.numberOfHiddenLayers+1)]

    def forward(self, X):
        """Propagate inputs through network"""

        # self.z = np.zeros(self.numberOfHiddenLayers+1)
        # self.a = np.zeros(self.numberOfHiddenLayers+1)
        self.z = [0 for i in range(self.numberOfHiddenLayers+1)]
        self.a = [0 for i in range(self.numberOfHiddenLayers+1)]
        
        for i in range(self.numberOfHiddenLayers+1):

            # first layer
            if i == 0:
                # z[] = np.dot(self.X, W) + B  # NB! numpy turns the vector B into a matrix of the correct shape so that it can be added to the other matrix
                # print(len(self.B))
                self.z[0] = np.dot(X, self.W[0]) + self.B[0]
                self.a[0] = self.sigmoid(self.z[0])

            # hidden layers
            elif 0 < i <= self.numberOfHiddenLayers:
                self.z[i] = np.dot(self.a[i-1], self.W[i]) + self.B[i]
                self.a[i] = self.sigmoid(self.z[i]) 
            
            # elif i == self.numberOfHiddenLayers:
            #     self.z[i] = np.dot(self.a[-2], self.W[-1]) + self.B[-1].flatten()
            #     self.a[i] = self.sigmoid(self.z[i]) 

        # self.z2 = np.dot(self.X, self.W1) + self.B1 
        # self.a2 = self.sigmoid(self.z2)
        # self.z3 = np.dot(self.a2, self.W2) + self.B2
        # yHat = self.sigmoid(self.z3) 
            

        yHat = self.a[-1]
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
    
    def costFunction(self, X, y): 
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        regularization = sum([np.linalg.norm(self.W[i]) for i in range(len(self.W))])
        
        C = 0.5*sum((y-self.yHat)**2) + self.lmd*regularization
        return C
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)

        delta = [0 for i in range(self.numberOfHiddenLayers+1)]
        dJdW = [0 for i in range(self.numberOfHiddenLayers+1)]
        dJdB = [0 for i in range(self.numberOfHiddenLayers+1)]

        for i in range(self.numberOfHiddenLayers, 0, -1):
            delta[i] = np.multiply(-(y-self.a[i]), self.sigmoidPrime(self.z[i]))
            dJdW[i] = np.dot(self.a[i-1].T, delta[i])    
            dJdB[i] = np.sum(delta[i], axis=0)

        delta[0] = np.multiply(-(y-self.a[0]), self.sigmoidPrime(self.z[0]))
        dJdW[0] = np.dot(self.X.T, delta[0])    
        dJdB[0] = np.sum(delta[0], axis=0)
        
        # delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        # dJdW2 = np.dot(self.a2.T, delta3)
        
        # delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        # dJdW1 = np.dot(X.T, delta2)  
        
        # dJdB1 = np.sum(delta2, axis=0)
        # dJdB2 = np.sum(delta3, axis=0)
        
        # return dJdW1, dJdW2, dJdB1, dJdB2
        return dJdW, dJdB

    def backpropagate(self):
        # dJdW1, dJdW2, dJdB1, dJdB2 = self.costFunctionPrime(self.trainX, self.trainY)
        dJdW, dJdB = self.costFunctionPrime(self.trainX, self.trainY)

        if (self.method=='GD'):
            for i in range(len(self.W)):
                self.W[i] = self.W[i] - self.eta * dJdW[i] - self.momentum*self.change[i] 
                self.B[i] = self.B[i] - self.eta * dJdB[i]
                self.change[i] = self.eta * dJdW[i] + self.momentum*self.change[i]

        # """ HER KAN VI IMPLEMENTERE FLERE GD METODER """
        # if (self.method=='GD'):
        #     self.W1 = self.W1 - self.eta * dJdW1 - self.momentum*self.change1 
        #     self.W2 = self.W2 - self.eta * dJdW2 - self.momentum*self.change2
            
        #     self.B1 = self.B1 - self.eta * dJdB1
        #     self.B2 = self.B2 - self.eta * dJdB2

        #     self.change1 = self.eta * dJdW1 + self.momentum*self.change1
        #     self.change2 = self.eta * dJdW2 + self.momentum*self.change2
        
        # elif (self.method="SGD"):
        #     SGD() 
        
        else:
            print('Gradient descent method not recognised :(')
            exit()

    def train(self, trainX, trainY, testX, testY, method='GD'):
        self.method = method

        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY

        # Make empty list to store training costs:
        self.J = []
        self.testJ = []

        self.J.append(self.costFunction(trainX, trainY))
        self.testJ.append(self.costFunction(testX, testY))

        for i in range(self.maxIterations):
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
    # np.random.seed(2)
    
    # Setting the data
    n = 1000
    x = 2*np.random.rand(n,1)-1

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
    nodes = np.array([10, 10, 10])
    NN = Neural_Network(X_train, 3, nodes, outputLayerSize=1, eta=0.01, lmd=0, momentum=0.01, maxIterations=500)
    NN.train(X_train, y_train, X_test, y_test, method='GD')

    """
    # Training the network
    NN = Neural_Network()
    NN.train(X_train, y_train, X_test, y_test, method='GD')
    """
    # Plotting results
    plt.figure(figsize=(6,4))
    plt.plot(NN.J)
    plt.plot(NN.testJ)
    plt.grid()
    plt.title('Training our neural network')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()
    
    x = np.linspace(-1,1,n).reshape(1,-1).T

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
    
