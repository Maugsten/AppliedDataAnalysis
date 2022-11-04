"""
Code is loosely based on Stephen Welch's 'Neural Networks Demystified'-series. 
https://github.com/stephencwelch/Neural-Networks-Demystified
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Neural_Network(object):
    def __init__(self, X, hidden_layers=1, nodes=10, nodes_output=1, eta=0.01, lmd=0, momentum=0, max_iterations=500):        
        # Define hyperparameters
        # self.inputLayerSize = 1
        # self.outputLayerSize = 1
        # self.hiddenLayerSize = 10
        self.X = X
        self.features = np.shape(self.X)[1]  # number of columns/features
        self.hidden_layers = hidden_layers
        self.nodes = nodes
        self.nodes_output = nodes_output  # number of nodes in output layer

        ## initial Weights and biases for each layer ##
        np.random.seed(2)
        prev_num_nodes = self.nodes
        self.W_list = []
        self.B_list = []

        # create weights and biases for each layer
        for i in range(self.hidden_layers+1):

            # first layer
            if i == 0:
                # matrix of weights (features x nodes)
                W = np.random.randn(self.features,self.nodes)
                self.W_list.append(W)
                # column vector of biases 
                B = np.random.randn(self.nodes,1)
                self.B_list.append(B)

            # hidden layers
            elif 0 < i < self.hidden_layers:
                # random number of nodes between 2 and 10
                rand_num_nodes = np.random.randint(2,11)
                # matrix of weights (previous number of nodes x random number of nodes)
                W = np.random.randn(prev_num_nodes, rand_num_nodes)
                self.W_list.append(W)
                # column vector of biases (one bias per node)
                B = np.random.randn(rand_num_nodes,1)
                self.B_list.append(B)
                prev_num_nodes = rand_num_nodes 
            
            # output layer
            elif i == self.hidden_layers:
                W = np.random.randn(prev_num_nodes, self.nodes_output)
                self.W_list.append(W)
                B = np.random.randn(self.nodes_output,1)
                self.hidden_layer_biases.append(B)


        # self.W1 = np.random.randn(self.features,self.nodes)  # features x nodes, first layer. This shape means that we don't have to transpose the matrix
        # self.W2 = np.random.randn(self.features,self.nodes) # second layer
        # self.B1 = np.zeros(self.hiddenLayerSize) + .01  # i.e. 10 zeros
        # self.B2 = np.zeros(self.outputLayerSize) + .01  # i.e. 1 zero

        # More hyperparameters
        self.eta = eta              # Learning rate
        self.momentum = momentum    # Momentum
        self.lmd = lmd              # Regularization parameter
        self.max_iterations = max_iterations
        self.change1 = 0
        self.change2 = 0

    def forward(self):
        """Propagate inputs through network"""

        z_prev = 0
        for i in range(self.hidden_layers+1):
            W = self.W_list[i]  # weight matrix for layer i
            B = self.B_list[i]  # bias vector for layer i

            # first layer
            if i == 0:
                z = np.dot(self.X, W) + B  # NB! numpy turns the vector B into a matrix of the correct shape so that it can be added to the other matrix
                z_prev = z

            # hidden layers
            elif 0 < i < self.hidden_layers:
                z = np.dot(self.X, W) + B

            elif i == self.hidden_layers:
                z

        self.z2 = np.dot(self.X, self.W1) + self.B1 
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
        
        dJdB1 = np.sum(delta2, axis=0)
        dJdB2 = np.sum(delta3, axis=0)
        
        return dJdW1, dJdW2, dJdB1, dJdB2

    def backpropagate(self):
        dJdW1, dJdW2, dJdB1, dJdB2 = self.costFunctionPrime(self.trainX, self.trainY)

        """ HER KAN VI IMPLEMENTERE FLERE GD METODER """
        if (self.method=='GD'):
            self.W1 = self.W1 - self.eta * dJdW1 - self.momentum*self.change1 
            self.W2 = self.W2 - self.eta * dJdW2 - self.momentum*self.change2
            
            self.B1 = self.B1 - self.eta * dJdB1
            self.B2 = self.B2 - self.eta * dJdB2

            self.change1 = self.eta * dJdW1 + self.momentum*self.change1
            self.change2 = self.eta * dJdW2 + self.momentum*self.change2
        
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
    y = 3 + 2*x + 3*x**2 #+ noise
    # y = np.e**(-x**2) #+ noise

    # Scaling the data
    x = x/np.max(x)
    y = y/np.max(y) #Max test score is 100

    print(np.shape(x))
    print(np.shape(y))

    # Splitting into train and test data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Training the network
    NN = Neural_Network()
    NN.train(X_train, y_train, X_test, y_test, method='GD')

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
