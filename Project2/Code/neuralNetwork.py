
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from NN_gradient_descent_methods import *
from project1_functions import *


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

        # for stochastic gradinet descent
        self.epochs = epochs        
        self.batchSize = batchSize
        self.numOfBatches = int(np.shape(self.X)[0] / self.batchSize)
        

    def forward(self, X):
        """Propagate inputs through network.
        
        Args:
            - X (ndarray): input data

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

        yHat = self.a[-1]
        return yHat
        
    def sigmoid(self, z):
        """ The standard logistic function
        
        Args: 
            - z (ndarray): output from layer, input for activation function 
        """
       
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self, z):
        """ The derivative of the standard logistic function.
        
        Args: 
            - z (ndarray): output from layer, input for activation function 

        Returns:
            - gradient of the sigmoid
        """
        
        return np.exp(-z)/((1+np.exp(-z))**2)

    def ReLU(self, z):
        """ The ReLU function.
        
        Args:
            - z (ndarray): output from layer, input for activation function 
        """

        return np.maximum(0, z)  # element-wise maximum of arrays elements, takes the maximum of 0 and the z-value
    
    def ReLUPrime(self, z):
        """ Derivative of ReLU
        
        Args:
            - z (ndarray): output from layer, input for activation function 
        """

        return np.where(z <= 0, 0, 1)  # element-wise maximum of arrays elements, takes the maximum of 0 and the z-value

    def leakyReLU(self, z, a=0.01):
        """ The leaky ReLU function. 
        
        Args:
            - z (ndarray): output from layer, input for activation function 
            - a (optional, float): leaky ReLU parameter

        """

        return np.where(z <= 0, a * z, z)  # where z <= 0, yield a*z, otherwise yield z

    def leakyReLUPrime(self, z, a=0.01):
        """ Derivative of leakyReLU
        
        Args:
            - z (ndarray): output from layer, input for activation function 
            - a (optional, float): leaky ReLU parameter

        """

        return np.where(z <= 0, a, 1)  # where z > 0, yield 1, otherwise yield a
    
    def costFunction(self, X, y): 
        """ Calculates the cost for a given model. 

        Args:
            - X (ndarray): input data
            - y (ndarray): data to be measured against (either training or test data)

        Returns:
            - C (float): the cost of the model       
        """
        
        # prediction value
        self.yHat = self.forward(X)
        regularization = sum([np.linalg.norm(self.W[i]) for i in range(len(self.W))])
        
        C = 0.5*sum((y-self.yHat)**2) + self.lmd*regularization
        return C
        
    def costFunctionPrime(self, X, y):
        """ Computes the gradients in the network. 

        Args:
            - X (ndarray): input data
            - y (ndarray): data to be measured against (either training or test data)

        Returns:
            - dCdW (nested list): nested list of gradients in terms of weights. Every element contains a matrix of gradients
            - dCdB (nested list): nested list of gradients in terms of biases. Every element contains a vector of gradients 
        """
        #Compute derivative with respect to weights and biases for a given X and y:
        self.yHat = self.forward(X)

        delta = [0 for i in range(self.numberOfHiddenLayers+1)]
        dCdW = [0 for i in range(self.numberOfHiddenLayers+1)]  # list of derivatives of the cost function in terms of weights
        dCdB = [0 for i in range(self.numberOfHiddenLayers+1)]  # list of derivatives of the cost function in terms of biases

        delta[-1] = self.a[-1] - y
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
            - trainX (ndarray): training data
            - trainY (ndarray): training data to fit

        Returns: None
        """
        dCdW, dCdB = self.costFunctionPrime(trainX, trainY)

        if self.method == 'GD':
            updatedW, updatedB = gradient_descent(dCdW, dCdB, self.W, self.B, self.eta, self.momentum, self.change, self.optimizer, self.RMS_W, self.RMS_B, self.M_W, self.M_B, self.iteration, self.dCdW2, self.dCdB2)
            self.W = updatedW
            self.B = updatedB

        elif self.method == 'SGD':
            # NB: the gradient descent function is the same for both GD and SGD, the stochastic part happens in the neural network 
            updatedW, updatedB = gradient_descent(dCdW, dCdB, self.W, self.B, self.eta, self.momentum, self.change, self.optimizer, self.RMS_W, self.RMS_B, self.M_W, self.M_B, self.epoch, self.dCdW2, self.dCdB2)
            self.W = updatedW
            self.B = updatedB
        
        else:
            print('Gradient descent method not recognised :(')
            exit()

    def train(self, trainX, trainY, testX, testY, method='GD', optimizer='None'):
        """ Trains and tests the nerual network.

        Args:
            - trainX (ndarray): training data 
            - trainY (ndarray): training data to fit
            - testX (ndarray): test data
            - testY (ndarray): test data to fit
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
            
        # for AdaGrad
        dCdW2 = [0 for i in range(self.numberOfHiddenLayers+1)]
        dCdB2 = [0 for i in range(self.numberOfHiddenLayers+1)]

        # for RMSProp and Adam
        RMS_W = [0 for i in range(self.numberOfHiddenLayers+1)]
        RMS_B = [0 for i in range(self.numberOfHiddenLayers+1)]

        # for Adam
        M_W = [0 for i in range(self.numberOfHiddenLayers+1)]
        M_B = [0 for i in range(self.numberOfHiddenLayers+1)]

        # gradient descent
        if self.method == 'GD':

            self.dCdW2 = dCdW2
            self.dCdB2 = dCdB2
            self.RMS_W = RMS_W
            self.RMS_B = RMS_B
            self.M_W = M_W
            self.M_B = M_B

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

                # resetting these parameters between each epoch to avoid overflow error
                self.dCdW2 = dCdW2
                self.dCdB2 = dCdB2
                self.RMS_W = RMS_W
                self.RMS_B = RMS_B
                self.M_W = M_W
                self.M_B = M_B

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
            - y (array): the data we are fitting
        
        Returns:
            - mse (float): mean square error
        """
        mse = np.mean(np.mean((y - self.yHat)**2, axis=1, keepdims=True))
        return mse
    

if __name__=="__main__":
    """ ========================= Regression ========================= """

    """ In this section, we set up the data we want to analyse. This includes computing, reshaping and scaling the data. """
    print("Score to beat:  ", 0.00768127)

    # Sets seed so results can be reproduced.
    np.random.seed(2000)  

    # Defines domain. No need to scale this data as it's already in the range (0,1)
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y) 

    # Standard deviation of the noise
    sigma = .1  

    mseF = 0
    reps = 1
    for i in range(int(reps)):
        # Franke function with stochastic noise
        z = FrankeFunction(x, y) + np.random.normal(0, sigma, x.shape)
        maxVal = np.amax(z)
        minVal = np.amin(z)
        z = (z-minVal)/(maxVal-minVal)
        m, n = np.shape(z)
        z_ = z.reshape(-1,1)

        # This is just to make some error prediction
        zF = FrankeFunction(x, y)
        zF = (zF-minVal)/(maxVal-minVal)
        zF_ = zF.reshape(-1,1)
        mseF += np.mean(np.mean((z - zF)**2, axis=1, keepdims=True))
    print('Mean error with no-noise Franke: ', mseF/reps)

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
    nodes = np.array([25])
    NN = Neural_Network(X_train, len(nodes), nodes, outputLayerSize=1, eta=0.001, lmd=0, momentum=0, maxIterations=500, epochs=100, batchSize=5)
    NN.train(X_train, z_train, X_test, z_test, method='SGD', optimizer='Adam')

    mse = NN.MSE(z_test)
    print("Neural Network: ", mse)

    

    """ In this section, we vary the number of hidden layers and nodes. """
    # nLayers = 3
    # nNodes = [5, 10, 15, 20, 25, 30]
    # MSE = np.zeros((nLayers, len(nNodes)))

    # for i in range(nLayers):
    #     for j in range(len(nNodes)):
    #         nodes = np.array([nNodes[j] for e in range(i+1)])
    #         print(nodes)
    #         NN = Neural_Network(X_train, len(nodes), nodes, outputLayerSize=1, eta=0.01, lmd=0, momentum=0, maxIterations=500, epochs=100, batchSize=10)
    #         NN.train(X_train, z_train, X_test, z_test, method='SGD')
    #         MSE[i,j] = NN.MSE(z_test)

    # print(MSE)

    # fig, ax = plt.subplots(figsize=(6,4))
    # im = ax.imshow(MSE, cmap="YlGn")

    # cbar = ax.figure.colorbar(im, ax=ax)
    # cbar.ax.set_ylabel("", rotation=-90, va="bottom")
    
    # layers = [i+1 for i in range(nLayers)]
    # ax.set_yticks(np.arange(len(layers)), labels=layers)
    # ax.set_xticks(np.arange(len(nNodes)), labels=nNodes)
    # for i in range(len(layers)):
    #     for j in range(len(nNodes)):
    #         text = ax.text(j, i, int(1e4*MSE[i, j])/1e4,
    #                     ha="center", va="center")
    # ax.set_title("MSE of Neural Network Prediction")
    # ax.set_ylabel("Number of Hidden Layers")
    # ax.set_xlabel("Number of Nodes in Layers")
    # fig.tight_layout()
    # plt.show()

    """ In this section, we vary a fixed learning rate """
    # nodes = [15]
    # etas = np.logspace(-.1, -5, 30)
    # MSE = np.zeros(len(etas))
    # for i in range(len(etas)):
    #     print(etas[i])
    #     NN = Neural_Network(X_train, len(nodes), nodes, outputLayerSize=1, eta=etas[i], lmd=0, momentum=0, maxIterations=500)#, epochs=3, batchSize=10)
    #     NN.train(X_train, z_train, X_test, z_test, method='GD', optimizer='Adam')
    #     MSE[i] = NN.MSE(z_test)

    # plt.figure(figsize=(6,4))
    # plt.plot(etas, MSE, 'r', linewidth=1.5)
    # plt.xscale('log')
    # # plt.yscale('log')
    # plt.title('MSE of Neural Network Prediction')
    # plt.xlabel('Learning Rate')
    # plt.ylabel('MSE')
    # plt.grid()
    # plt.show()

    """ In this section, we vary momentum """
    # nodes = [20]
    # # momentums = np.logspace(-.5, -3, 4)
    # momentums = [0.5, 0.1, 0.01, 0]
    
    # NN = Neural_Network(X_train, len(nodes), nodes, outputLayerSize=1, eta=0.01, lmd=0, momentum=momentums[0], maxIterations=500, epochs=100, batchSize=5)
    # NN.train(X_train, z_train, X_test, z_test, method='SGD')
    # C1 = NN.C

    # NN = Neural_Network(X_train, len(nodes), nodes, outputLayerSize=1, eta=0.01, lmd=0, momentum=momentums[1], maxIterations=500, epochs=100, batchSize=5)
    # NN.train(X_train, z_train, X_test, z_test, method='SGD')
    # C2 = NN.C

    # NN = Neural_Network(X_train, len(nodes), nodes, outputLayerSize=1, eta=0.01, lmd=0, momentum=momentums[2], maxIterations=500, epochs=100, batchSize=5)
    # NN.train(X_train, z_train, X_test, z_test, method='SGD')
    # C3 = NN.C

    # NN = Neural_Network(X_train, len(nodes), nodes, outputLayerSize=1, eta=0.01, lmd=0, momentum=momentums[3], maxIterations=500, epochs=100, batchSize=5)
    # NN.train(X_train, z_train, X_test, z_test, method='SGD')
    # C4 = NN.C

    # C1 = np.array(C1).flatten()
    # C2 = np.array(C2).flatten()
    # C3 = np.array(C3).flatten()
    # C4 = np.array(C4).flatten()

    # # Smoothing was not used in final results
    # kernel_size = 1
    # kernel = np.ones(kernel_size) / kernel_size

    # C1Smooth = np.convolve(C1[:100], kernel, mode='same')
    # C2Smooth = np.convolve(C2[:100], kernel, mode='same')
    # C3Smooth = np.convolve(C3[:100], kernel, mode='same')
    # C4Smooth = np.convolve(C4[:100], kernel, mode='same')

    # plt.figure(figsize=(6,4))
    # plt.plot(C1Smooth, label='Momentum=0.5')
    # plt.plot(C2Smooth, label='Momentum=0.1')
    # plt.plot(C3Smooth, label='Momentum=0.01')
    # plt.plot(C4Smooth, label='Momentum=0')
    # plt.grid()
    # # plt.yscale('log')
    # plt.title('Cost Function per Iteration')
    # plt.xlabel('Iterations')
    # plt.ylabel('Cost')
    # plt.legend()

    # plt.figure(figsize=(6,4))
    # plt.plot(C1Smooth, label='Momentum=0.5')
    # plt.plot(C2Smooth, label='Momentum=0.1')
    # plt.plot(C3Smooth, label='Momentum=0.01')
    # plt.plot(C4Smooth, label='Momentum=0')
    # plt.grid()
    # plt.yscale('log')
    # plt.title('Cost Function per Iteration')
    # plt.xlabel('Iterations')
    # plt.ylabel('Cost')
    # plt.legend()    
    # plt.show()

    """ Activation functions """

    # tick = time.time()

    # reps = 1
    # mse = 0
    # for i in range(reps):
    #     nodes = np.array([50, 50, 50])
    #     NN = Neural_Network(X_train, len(nodes), nodes, outputLayerSize=1, eta=0.01, lmd=0, momentum=0, maxIterations=500, epochs=100, batchSize=15)
    #     NN.train(X_train, z_train, X_test, z_test, method='SGD')

    #     mse += NN.MSE(z_test)

    # tock = time.time()
    # seconds = tock - tick
    # print('\nSeconds: ', np.floor(seconds), 's\n')
    # print("Neural Network: ", mse/reps)

    """ Adaptive Learning """
    # nodes = np.array([6])
    # NN = Neural_Network(X_train, len(nodes), nodes, outputLayerSize=1, eta=0.001, lmd=0.0, momentum=0, maxIterations=1000, epochs=100, batchSize=5)
    # NN.train(X_train, z_train, X_test, z_test, method='SGD')
    # costTrainNoLearning = NN.C
    # costTestNoLearning = NN.testC

    # # MSE
    # mse = NN.MSE(z_test)
    # print("Neural Network, SGD None: ", mse)

    # NN = Neural_Network(X_train, len(nodes), nodes, outputLayerSize=1, eta=0.001, lmd=0.0, momentum=0, maxIterations=1000, epochs=100, batchSize=5)
    # NN.train(X_train, z_train, X_test, z_test, method='SGD', optimizer="AdaGrad")
    # costTrainAdagrad = NN.C
    # costTestAdagrad = NN.testC

    # # MSE
    # mse = NN.MSE(z_test)
    # print("Neural Network, SGD AdaGrad: ", mse)

    # NN = Neural_Network(X_train, len(nodes), nodes, outputLayerSize=1, eta=0.001, lmd=0.0, momentum=0, maxIterations=1000, epochs=100, batchSize=5)
    # NN.train(X_train, z_train, X_test, z_test, method='SGD', optimizer="RMSprop")
    # costTrainRMSprop = NN.C
    # costTestRMSprop = NN.testC

    # # MSE
    # mse = NN.MSE(z_test)
    # print("Neural Network, SGD RMSProp: ", mse)

    # NN = Neural_Network(X_train, len(nodes), nodes, outputLayerSize=1, eta=0.001, lmd=0.0, momentum=0, maxIterations=1000, epochs=100, batchSize=5)
    # NN.train(X_train, z_train, X_test, z_test, method='SGD', optimizer="Adam")
    # costTrainAdam = NN.C
    # costTestAdam = NN.testC

    # # Plotting results
    # plt.figure(figsize=(6,4))
    # plt.plot(costTrainNoLearning, label='No Adaptive Learning')
    # plt.plot(costTrainAdagrad, label='Adagrad')
    # plt.plot(costTrainRMSprop, label='RMSprop')
    # plt.plot(costTrainAdam, label='Adam')
    # plt.grid()
    # plt.title('Training Data Cost')
    # plt.xlabel('Iterations')
    # plt.ylabel('Cost')
    # plt.legend()

    # plt.figure(figsize=(6,4))
    # plt.plot(costTestNoLearning, label='No Adaptive Learning')
    # plt.plot(costTestAdagrad, label='Adagrad')
    # plt.plot(costTestRMSprop, label='RMSprop')
    # plt.plot(costTestAdam, label='Adam')
    # plt.grid()
    # plt.title('Test Data Cost')
    # plt.xlabel('Iterations')
    # plt.ylabel('Cost')
    # plt.legend()
    # plt.show()

    """ In this section, we make surface plots. """
    # nodes = np.array([25])
    # NN = Neural_Network(X_train, len(nodes), nodes, outputLayerSize=1, eta=0.001, lmd=0, momentum=0, maxIterations=1000, epochs=100, batchSize=5)
    # NN.train(X_train, z_train, X_test, z_test, method='SGD', optimizer='Adam')
    
    # # MSE
    # mse = NN.MSE(z_test)
    # print("Neural Network: ", mse)

    
    # zHat = NN.forward(X).reshape((m,n))
    # fig = plt.figure(figsize=plt.figaspect(0.5), constrained_layout=True)
    # ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    # ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # surf1 = ax1.plot_surface(x, y, z, cmap=cm.coolwarm,
    #                         linewidth=0, antialiased=False)
    # surf2 = ax2.plot_surface(x, y, zHat, cmap=cm.coolwarm,
    #                         linewidth=0, antialiased=False)

    # ax1.set_zlim(np.amin(z), np.amax(z))
    # ax1.zaxis.set_major_locator(LinearLocator(10))
    # ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # ax1.set_xlabel('x')
    # ax1.set_ylabel('y')
    # ax1.yaxis._axinfo['label']
    # ax1.set_zlabel('z')
    # ax1.set_title("Original Data")

    # ax2.set_zlim(np.amin(z), np.amax(z))
    # ax2.zaxis.set_major_locator(LinearLocator(10))
    # ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # ax2.set_xlabel('x')
    # ax2.set_ylabel('y')
    # ax2.yaxis._axinfo['label']
    # ax2.set_zlabel('z')
    # ax2.set_title("Neural Network Fit")
    # fig.colorbar(surf1, shrink=0.5, aspect=10)

    # # Plotting results
    # plt.figure(figsize=(6,4))
    # plt.plot(NN.C, label='Train Data')
    # plt.plot(NN.testC, label='Test Data')
    # plt.grid()
    # plt.title('Training our neural network')
    # plt.xlabel('Iterations')
    # plt.ylabel('Cost')
    # plt.legend()
    # plt.show()



    """ THIS IS THE WINNER SO FAR! MSE=0.005246113511425173 """
    # nodes = np.array([70, 70, 70, 70, 70, 70])
    # NN = Neural_Network(X_train, len(nodes), nodes, outputLayerSize=1, eta=0.01, lmd=0, momentum=0, maxIterations=1000, epochs=100, batchSize=5)
    # NN.train(X_train, z_train, X_test, z_test, method='SGD')#, method='SGD', optimizer='Adam')



    """ ========================= Classification ========================= """
    """ Setting up data """
    # M = Malignant = BAD
    # B = Benign = NOT BAD   
    num_lines = sum(1 for line in open('data.csv', 'r'))
    file = open('data.csv', 'r')
    features = np.array(file.readline().split(','))[:-1]
    data = []
    for i in range(num_lines-1):
        data.append(np.array(file.readline().split(',')))
    data = np.array(data)

    X = data[:,2:]
    X = X.astype(np.float64)
    y = data[:,1]
    for i in range(len(y)):
        if y[i]=="M":
            y[i]=1
        elif y[i]=="B":
            y[i]=0
        else:
            print('Unrecognised data')
    y = y.astype(np.float64).reshape(1,-1).T
    # Scaling the features (normalization)
    for i in range(len(X[0,:])):
        X[:,i] = (X[:,i]-np.amin(X[:,i])) / (np.amax(X[:,i]) - np.amin(X[:,i]))

    # Note to self: shape of X = (569, 30)

    # Splitting into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    """ Basic Network """
    # Training the network
    nodes = np.array([25])
    NN = Neural_Network(X_train, len(nodes), nodes, outputLayerSize=1, eta=0.01, lmd=0, momentum=0, maxIterations=1000, epochs=100, batchSize=5)
    NN.train(X_train, y_train, X_test, y_test, method='GD', optimizer='Adam')#, method='SGD', optimizer='Adam')

    prediction = NN.forward(X_test)
    for i in range(len(prediction)):
        if prediction[i]>=0.5:
            prediction[i]=1
        elif prediction[i]<0.5:
            prediction[i]=0
        else:
            print('Unrecognised data')

    errors = abs(y_test-prediction)
    accuracy = 1 - np.sum(errors)/len(errors)
    print('Accuracy: {}'.format(accuracy))

    # Plotting results
    plt.figure(figsize=(6,4))
    plt.plot(NN.C)
    plt.plot(NN.testC)
    plt.grid()
    plt.title('Training our neural network')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()

    """ Layers and nodes """
    # nLayers = 3
    # nNodes = [5, 10, 15, 20, 25, 30]
    # acc = np.zeros((nLayers, len(nNodes)))

    # for i in range(nLayers):
    #     for j in range(len(nNodes)):
    #         nodes = np.array([nNodes[j] for e in range(i+1)])
    #         print(nodes)
    #         NN = Neural_Network(X_train, len(nodes), nodes, outputLayerSize=1, eta=0.01, lmd=0, momentum=0, maxIterations=500, epochs=100, batchSize=10)
    #         NN.train(X_train, y_train, X_test, y_test, method='SGD')
    #         prediction = NN.forward(X_test)
    #         for k in range(len(prediction)):
    #             if prediction[k]>=0.5:
    #                 prediction[k]=1
    #             elif prediction[k]<0.5:
    #                 prediction[k]=0
    #             else:
    #                 print('Unrecognised data')

    #         errors = abs(y_test-prediction)
    #         accuracy = 1 - np.sum(errors)/len(errors)
    #         acc[i,j] = accuracy

    # fig, ax = plt.subplots()
    # im = ax.imshow(acc, cmap="PuBu")

    # cbar = ax.figure.colorbar(im, ax=ax)
    # cbar.ax.set_ylabel("", rotation=-90, va="bottom")
    
    # layers = [i+1 for i in range(nLayers)]
    # ax.set_yticks(np.arange(len(layers)), labels=layers)
    # ax.set_xticks(np.arange(len(nNodes)), labels=nNodes)
    # for i in range(len(layers)):
    #     for j in range(len(nNodes)):
    #         text = ax.text(j, i, np.floor(1e3*acc[i, j])/1e3,
    #                     ha="center", va="center")
    # ax.set_title("Accuracy of Neural Network Prediction")
    # ax.set_ylabel("Number of Hidden Layers")
    # ax.set_xlabel("Number of Nodes in Layers")
    # fig.tight_layout()
    # plt.show()