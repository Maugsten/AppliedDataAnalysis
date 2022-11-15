from fileinput import filename
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neural_network
from sklearn.model_selection import train_test_split


class Neural_Network(object):
    def __init__(self, eta=0.01, lmd=0, momentum=0, max_iterations=500):        
        # Define hyperparameters
        self.inputLayerSize = 30
        self.outputLayerSize = 1
        self.hiddenLayerSize = 10
        
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
        self.change1 = 0
        self.change2 = 0

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
        #Compute derivative with respect to W1 and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        # print(np.shape(delta3))
        # print(np.shape(self.W2.T))
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
    """ Reading and pre-processing data """
    # M = Malignant = BAD
    # B = Benign = NOT BAD   
    num_lines = sum(1 for line in open('archive\data.csv', 'r'))
    file = open('archive\data.csv', 'r')
    features = np.array(file.readline().split(','))[:-1]
    data = []
    for i in range(num_lines-1):
        data.append(np.array(file.readline().split(',')))
    data = np.array(data)
    X = data[:,2:]
    X = X.astype(np.float64)
    y = data[:,1]
    for i in range(len(y)):
        if y[i]=='M':
            y[i]=1
        elif y[i]=='B':
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

    # Training the network
    NN = Neural_Network()
    NN.train(X_train, y_train, X_test, y_test, method='GD')

    prediction = NN.forward(X)
    for i in range(len(prediction)):
        if prediction[i]>=0.5:
            prediction[i]=1
        elif prediction[i]<0.5:
            prediction[i]=0
        else:
            print('Unrecognised data')

    errors = abs(y-prediction)
    accuracy = 1 - np.sum(errors)/len(errors)
    print('Accuracy: {}'.format(accuracy))

    # Plotting results
    plt.figure(figsize=(6,4))
    plt.plot(NN.J)
    plt.plot(NN.testJ)
    plt.grid()
    plt.title('Training our neural network')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()