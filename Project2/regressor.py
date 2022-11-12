"""
Code is loosely based on Stephen Welch's 'Neural Networks Demystified'-series. 
https://github.com/stephencwelch/Neural-Networks-Demystified
"""

from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from NN_gradient_descent_methods import *
from functions import *


class Neural_Network(object):
    def __init__(self, X, numberOfHiddenLayers, nodes, outputLayerSize=1, eta=0.01, lmd=0, momentum=0, maxIterations=1000, epochs=300, batchSize=10):        
        
        # define hyperparameters
        self.X = X  # feature data
        self.inputLayerSize = np.shape(self.X)[1] # number of features
        self.outputLayerSize = outputLayerSize    # number of nodes in output layer
        self.numberOfHiddenLayers = numberOfHiddenLayers  # number of hidden layers
        self.nodes = nodes  # array with number of nodes in each hidden layer
        
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

                # column vector of biases (one bias per node)
                b = np.random.randn(self.nodes[0])
                self.B.append(b)

            # hidden layers
            elif 0 < i < self.numberOfHiddenLayers:
                # matrix of weights (previous number of nodes x current number of nodes)
                w = np.random.randn(self.nodes[i-1], self.nodes[i])
                self.W.append(w)

                # column vector of biases (one bias per node)
                b = np.random.randn(self.nodes[i])
                self.B.append(b)
            
            # output layer
            elif i == self.numberOfHiddenLayers:
                # matrix of weights (previous number of nodes x number of nodes for output layer)
                w = np.random.randn(self.nodes[-1], self.outputLayerSize)
                self.W.append(w)
                
                # column vector of biases (one bias per node), usually one node in output layer which means one bias
                b = np.random.randn(self.outputLayerSize)
                self.B.append(b)

        # More hyperparameters
        self.eta = eta              # learning rate
        self.momentum = momentum    # momentum
        self.lmd = lmd              # regularization parameter
        self.maxIterations = maxIterations  # number of iterations through the network
        self.change = [0 for i in range(self.numberOfHiddenLayers+1)]  # list of a variable that is needed in the gradient descent methods

        self.z = [0 for i in range(self.numberOfHiddenLayers+1)]  # list to store the data after multiplying it with weights and adding biases
        self.a = [0 for i in range(self.numberOfHiddenLayers+1)]  # list to store the data after it has gone through an activation function

        # for AdaGrad optimizer method
        self.dCdW2 = [0 for i in range(self.numberOfHiddenLayers+1)]
        self.dCdB2 = [0 for i in range(self.numberOfHiddenLayers+1)]

        # for RMSprop and Adam optimizer methods
        self.RMS_W = [0 for i in range(self.numberOfHiddenLayers+1)]
        self.RMS_B = [0 for i in range(self.numberOfHiddenLayers+1)]

        # for Adam optimizer method
        self.M_W = [0 for i in range(self.numberOfHiddenLayers+1)]
        self.M_B = [0 for i in range(self.numberOfHiddenLayers+1)]

        # for stochastic gradinet descent
        self.epochs = epochs        
        self.batchSize = batchSize
        self.numOfBatches = int(np.shape(self.X)[0] / self.batchSize)
        

    def forward(self, X):
        """Propagate inputs through network.
        
        Args:
            - X (ndarray): input values

        Returns:
            - yHat (n x 1 array): predicted values
        """

        for i in range(self.numberOfHiddenLayers+1):

            # first layer
            if i == 0:
                self.z[0] = np.dot(X, self.W[0]) + self.B[0]  # NB! numpy turns the vector B into a matrix of the correct shape so that it can be added to the other matrix
                self.a[0] = self.sigmoid(self.z[0])

            # hidden layers
            elif 0 < i < self.numberOfHiddenLayers:
                self.z[i] = np.dot(self.a[i-1], self.W[i]) + self.B[i]
                self.a[i] = self.sigmoid(self.z[i])  

            # output layer
            elif i == self.numberOfHiddenLayers:
                self.z[i] = np.dot(self.a[i-1], self.W[i]) + self.B[i]
                self.a[i] = self.z[i]
                # self.a[i] = self.leakyReLU(self.z[i])  # for classification

        yHat = self.a[-1]
        return yHat
        
    def sigmoid(self, z):
        """ The sigmoid function
        
        Args: 
            - z (?): ?
        """
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self, z):
        """ Derivative of sigmoid.
        
        Args: 
            - z (?): ?

        Returns:
            - gradient of the sigmoid
        """
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)

    def ReLU(self, z):
        """ The ReLU function.
        
        Args:
            - z (?): ?
        """

        return np.maximum(0, z)  # element-wise maximum of arrays elements, takes the maximum of 0 and the z-value
    
    def ReLUPrime(self, z):
        """ Derivative of ReLU
        
        Args:
            - z (?): ?
        """

        return np.maximum(0, z)  # element-wise maximum of arrays elements, takes the maximum of 0 and the z-value

    def leakyReLU(self, z, a=0.01):
        """ The leaky ReLU function. 
        
        Args:
            - z (?): ?
            - a (optional, float): ?

        """

        return np.where(z <= 0, a * z, z)  # where z <= 0, yield a*z, otherwise yield z

    def leakyReLUPrime(self, z, a=0.01):
        """ Derivative of leakyReLU
        
        Args:
            - z (?): ?
            - a (optional, float): ?

        """

        return np.where(z > 0, 1, a)  # where z > 0, yield 1, otherwise yield a
    
    def costFunction(self, X, y): 
        """ Calculates the cost for a given model. 

        Args:
            - X (ndarray): 
            - y (ndarray): 

        Returns:
            - C (float): the cost of the model ?        
        """
        
        # prediction value
        self.yHat = self.forward(X)
        # penalty ?
        regularization = sum([np.linalg.norm(self.W[i]) for i in range(len(self.W))])
        
        C = 0.5*sum((y-self.yHat)**2) + self.lmd*regularization
        return C
        
    def costFunctionPrime(self, X, y):
        """ Computes the gradients in the network. 

        Args:
            - X (ndarray): 
            - y (ndarray): 

        Returns:
            - dCdW (nested list?): list of gradients in terms of weights. Every element contains a list of gradients for one layer in the network ?
            - dCdB (list?): list of gradients in terms of biases. Every element contains a gradient for one layer in the network ? 
        """
        #Compute derivative with respect to weights and biases for a given X and y:
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

        delta[0] = np.dot(delta[1], self.W[1].T)*self.sigmoidPrime(self.z[0])
        dCdW[0] = np.dot(X.T, delta[0])    
        dCdB[0] = np.sum(delta[0], axis=0) 

        return dCdW, dCdB

    def backpropagate(self, trainX, trainY):
        """ Updates the weights and biases of the neural network by using gradient descent methods.

        Args: 
            - trainX (ndarray): training data, independent variables
            - trainY (ndarray): training data to fit

        Returns: None
        """
        dCdW, dCdB = self.costFunctionPrime(trainX, trainY)

        if self.method == 'GD':
            updatedW, updatedB = gradient_descent(dCdW, dCdB, self.W, self.B, self.eta, self.momentum, self.change, self.optimizer, self.RMS_W, self.RMS_B, self.M_W, self.M_B, self.iteration, self.dCdW2, self.dCdB2)
            self.W = updatedW
            self.B = updatedB

        elif self.method == 'SGD':
            updatedW, updatedB = gradient_descent(dCdW, dCdB, self.W, self.B, self.eta, self.momentum, self.change, self.optimizer, self.RMS_W, self.RMS_B, self.M_W, self.M_B, self.epoch, self.dCdW2, self.dCdB2)
            self.W = updatedW
            self.B = updatedB
        
        else:
            print('Gradient descent method not recognised :(')
            exit()

    def train(self, trainX, trainY, testX, testY, method='GD', optimizer='None'):
        """ Trains and tests the nerual network.

        Args:
            - self ?
            - trainX (ndarray): training data, independent variables ? 
            - trainY (ndarray): training data to fit
            - testX (ndarray): test data, independent variables ?
            - testY (ndarray): test data, dependent variables
            - method (optional, str): choice of method, either GD (gradient descent, default) or SGD (stochastic gradient descent)
            - optimizer (optional, str): choice of optimizer, either None (plain gradient descent, default), AdaGrad, RMSprop or Adam 

        Returns: None
        """
        # method and optimizer
        self.method = method
        self.optimizer = optimizer
        
        # the splitted data
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY

        # lists for storing training and test costs:
        self.C = []
        self.testC = []

        self.C.append(self.costFunction(trainX, trainY))
        self.testC.append(self.costFunction(testX, testY))

        t0, t1 = 5, 50
        def learning_schedule(t):
            """ Tuning the learning rate.
            Args:
                - t (float)
            Returns: 
                - the learning rate
            """
            return t0/(t+t1)

        # gradient descent
        if self.method == 'GD':
            for iteration in range(self.maxIterations):
                self.iteration = iteration  # needed in the gradient descent function

                # updating weights and biases 
                self.backpropagate(self.trainX, self.trainY) 

                # storing the cost
                self.C.append(self.costFunction(trainX, trainY))
                self.testC.append(self.costFunction(testX, testY))

                # updating the learing rate
                self.eta = learning_schedule(iteration)


        # stochastic gradient descent
        if self.method == 'SGD':
            # lists for storing the batches
            Xbatches = [0 for i in range(self.numOfBatches)]
            Ybatches = [0 for i in range(self.numOfBatches)]

            for i in range(self.numOfBatches):
                # picking a random index
                random_index = self.batchSize * np.random.randint(self.numOfBatches)  
                # create minibatches
                Xbatches[i] = self.trainX[random_index:random_index + self.batchSize]
                Ybatches[i] = self.trainY[random_index:random_index + self.batchSize] 

            for epoch in range(self.epochs):
                self.epoch = epoch  # needed in the gradient descent method
                for i in range(self.numOfBatches):
                    # updating weights and biases 
                    self.backpropagate(Xbatches[i], Ybatches[i])

                    # storing the cost
                    self.C.append(self.costFunction(Xbatches[i], Ybatches[i]))
                    self.testC.append(self.costFunction(testX, testY))

                    # updating the learing rate
                    self.eta = learning_schedule(epoch + i)

    def MSE(self, y):
        """ Mean square error

        Args:
            - y
        """
        # mse = np.mean((y-self.yHat)**2)
        mse = np.mean(np.mean((y - self.yHat)**2, axis=1, keepdims=True))
        return mse
    
    def R2(self, y):
        """ R2 score
        
        Args:
            - y
        """
        r2 = 1 - sum((y-self.yHat)**2) / sum((y-np.mean(y))**2) 
        return r2


if __name__=="__main__":
    """ In this section, we set up the data we want to analyse. This includes computing, reshaping and scaling the data. """
    print("Score to beat:  ", 0.00768127)

    # Sets seed so results can be reproduced.
    # np.random.seed(1998)  

    # Defines domain. No need to scale this data as it's already in the range (0,1)
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y) 

    # Standard deviation of the noise
    sigma = .1  

    # Franke function with stochastic noise
    z = FrankeFunction(x, y) + np.random.normal(0, sigma, x.shape)
    m, n = np.shape(z)
    z_ = z.reshape(-1,1)

    # Feature matrix
    X = np.zeros((len(x)**2, 2))
    X[:,0] = x.flatten()
    X[:,1] = y.flatten()

    # Splitting into train and test data
    X_train, X_test, z_train, z_test = train_test_split(X, z_, test_size=0.2)

    # Scaling
    for j in range(len(X_test[0,:])):
        X_test[:,j] = (X_test[:,j] - np.amin(X_train[:,j])) / (np.amax(X_train[:,j]) - np.amin(X_train[:,j]))   # Normalization
        X_train[:,j] = (X_train[:,j] - np.amin(X_train[:,j])) / (np.amax(X_train[:,j]) - np.amin(X_train[:,j])) # Normalization
    z_test = (z_test - np.amin(z_train)) / (np.amax(z_train) - np.amin(z_train))   # Normalization
    z_train = (z_train - np.amin(z_train)) / (np.amax(z_train) - np.amin(z_train)) # Normalization

    """ Our basic network """
    nodes = np.array([50, 50, 50])
    NN = Neural_Network(X_train, len(nodes), nodes, outputLayerSize=1, eta=0.01, lmd=0, momentum=0, maxIterations=500, epochs=100, batchSize=5)
    NN.train(X_train, z_train, X_test, z_test, method='SGD')

    mse = NN.MSE(z_test)
    print("Neural Network: ", mse)

    

    """ In this section, we vary the number of hidden layers. """
    # n = 8
    # MSE = np.zeros(n)
    # for i in range(n):
    #     nodes = np.array([50 for j in range(i+1)])
    #     print(nodes)
    #     NN = Neural_Network(X_train, len(nodes), nodes, outputLayerSize=1, eta=0.01, lmd=0, momentum=0, maxIterations=500, epochs=100, batchSize=5)
    #     NN.train(X_train, z_train, X_test, z_test, method='SGD')
    #     MSE[i] = NN.MSE(z_test)
    
    # for i in range(len(MSE)):
    #     print(MSE[i])

    """ In this section, we vary the number of hidden layers. """
    nodeNumber = [1, 2, 3, 4, 5, 6, 7, 8]
    MSE = np.zeros(len(nodeNumber))
    for i in range(len(nodeNumber)):
        nodes = np.array([10*nodeNumber[i], 10*nodeNumber[i], 10*nodeNumber[i]])
        print(nodes)
        NN = Neural_Network(X_train, len(nodes), nodes, outputLayerSize=1, eta=0.01, lmd=0, momentum=0, maxIterations=500, epochs=100, batchSize=10)
        NN.train(X_train, z_train, X_test, z_test, method='SGD')
        MSE[i] = NN.MSE(z_test)
    
    for i in range(len(MSE)):
        print(MSE[i])


    """ In this section, we post-process our findings. """
    # MSE
    mse = NN.MSE(z_test)
    print("Neural Network: ", mse)

    
    zHat = NN.forward(X).reshape((m,n))
    fig = plt.figure(figsize=plt.figaspect(0.5), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    surf1 = ax1.plot_surface(x, y, z, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
    surf2 = ax2.plot_surface(x, y, zHat, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)

    ax1.set_zlim(np.amin(z), np.amax(z))
    ax1.zaxis.set_major_locator(LinearLocator(10))
    ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.yaxis._axinfo['label']
    ax1.set_zlabel('z')
    ax1.set_title("Original Data")

    ax2.set_zlim(np.amin(z), np.amax(z))
    ax2.zaxis.set_major_locator(LinearLocator(10))
    ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.yaxis._axinfo['label']
    ax2.set_zlabel('z')
    ax2.set_title("Neural Network Fit")
    fig.colorbar(surf1, shrink=0.5, aspect=10)

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



    """ THIS IS THE WINNER SO FAR! MSE=0.005246113511425173 """
    # nodes = np.array([70, 70, 70, 70, 70, 70])
    # NN = Neural_Network(X_train, len(nodes), nodes, outputLayerSize=1, eta=0.01, lmd=0, momentum=0, maxIterations=1000, epochs=100, batchSize=5)
    # NN.train(X_train, z_train, X_test, z_test, method='SGD')#, method='SGD', optimizer='Adam')



    """ OLD CODE
    np.random.seed(10)
    
    # Setting the data
    n = 1000
    x = np.random.rand(n,1)

    noise = np.random.normal(0, .5, x.shape)
    y = 3 + 2*x + 3*x**2 + 4*x**3 #+ noise
    # y = np.e**(-x**2) #+ noise
    # y = np.sin(x) + noise

    # Scaling the data
    x = x/np.max(x)
    y = y/np.max(y)  # max test score is 100

    # print(np.shape(x))
    # print(np.shape(y))

    X = np.zeros((len(x), 2))
    X[:,0] = x.reshape(1,-1)
    X[:,1] = x.reshape(1,-1)**2

    print(np.shape(X))
    print(np.shape(y))
    # Splitting into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(np.shape(X_train))
    print(np.shape(y_train))
    
    # nodes = np.array([5, 7, 4])
    nodes = np.array([7, 5])
    NN = Neural_Network(X_train, 2, nodes, outputLayerSize=1, eta=0.01, lmd=0.0001, momentum=0, epochs=1000)
    NN.train(X_train, y_train, X_test, y_test,method='SGD')#, method='SGD', optimizer='Adam')

    """
    # # Training the network
    # NN = Neural_Network()
    # NN.train(X_train, y_train, X_test, y_test, method='GD')
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

    y = 3 + 2*x + 3*x**2 + 4*x**3 #+ noise    
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
    """
