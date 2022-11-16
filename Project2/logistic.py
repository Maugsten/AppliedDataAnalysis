from cProfile import label
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  

class Classifier(object):
    def __init__(self, eta):        
        # Define hyperparameters
        self.eta = eta

    def logistic_function(self, x):
        return 1 / (1 + np.exp(-x))

    def optimize(self, X, y):
        self.W = np.random.rand(len(X[0,:]),1)
        n = 200

        # print(np.shape(X))
        # print(np.shape(self.W))
        # print(np.shape(y))

        for i in range(n):
            self.W = self.W - self.eta * (X.T @ ( self.logistic_function(X @ self.W) - y) )

        # self.W = self.W - self.eta *np.dot(X.T, self.sigmoid(np.dot(X, weights)) - np.reshape(y, (len(y), 1)) ) # - self.momentum*self.change

    def predict(self, X):
        yHat = self.logistic_function(X @ self.W)
        prediction = []
        for e in yHat:
            if e>0.5:
                prediction.append(1)
            else:
                prediction.append(0)
        return np.array(prediction).reshape(-1,1)

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

    # X,y = make_classification()
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.1)

    etas = np.logspace(-.5, -4, 1000)
    accuracyTrain = np.zeros_like(etas)
    accuracyTest = np.zeros_like(etas)
    for i in range(len(etas)):
        ModelMaker = Classifier(eta=etas[i])
        ModelMaker.optimize(X_tr, y_tr)

        pred_tr = ModelMaker.predict(X_tr)
        pred_te = ModelMaker.predict(X_te)

        errors = abs(y_tr - pred_tr)
        accuracy = 1 - np.sum(errors)/len(errors)
        # print('Train Accuracy: {}'.format(accuracy))
        accuracyTrain[i] = accuracy

        errors = abs(y_te - pred_te)
        accuracy = 1 - np.sum(errors)/len(errors)
        # print('Test Accuracy: {}'.format(accuracy))
        accuracyTest[i] = accuracy

    plt.figure(figsize=(6,4))
    plt.semilogx(etas, accuracyTrain, label="Train Data")
    plt.semilogx(etas, accuracyTest, label="Test Data")
    plt.grid()
    plt.title('Accuracy: Logistic Regression')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()