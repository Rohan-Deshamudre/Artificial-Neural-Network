import math
import numpy as np
import os.path

class Perceptron(object):

    """
    Parameters:
    -----------
    number_of_features: the number of features each input sample will have
    lr: learning rate (between 0 and 1)
    iterations: number of passes over training set
    Attributes:
    -----------
    weights: weights of each feature
    bias: the bias for each neuron
    misclassifications: number of errors in every iteration
    """
    def __init__(self, number_of_features, lr, iterations):
        self.lr = lr
        self.iterations = iterations
        # weights[0] = bias
        self.weights = np.random.uniform(-0.5, 0.5, 1 + number_of_features)
        self.weights[0] = 0
        self.misclassifications = []

    """
    X: input data set, the data array containing the 10 features of 7854 samples with shape = [n_samples][n_features]
    y: output data set, array of target features for the 7854 samples
    """
    def fit(self, X, y):

        path =  os.path.abspath(__file__ + ".\\..\\..\\resources\\"+"xor.txt")
        f = open(path, "a+")

        for i in range(self.iterations):
            totalError = 0
            for xi, target in zip(X, y):
                output = self.predict(xi)
                error = target - output
                totalError+=abs(error)
                delta = self.lr * xi * error

                self.weights[1:] += self.lr * xi * error
                self.misclassifications.append(error)

                print("xi: ", xi, " output: ", output, " weights: ", self.weights, " error: ", error, " delta: ", delta)
            meanError = totalError / 4
            f.write(str(meanError)+",")
        f.write("\n")
        f.close()
        return self

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.5, 1, 0)


if __name__ == "__main__":
    # OR
    # X = np.array([[0,0], [0,1], [1,0], [1,1]])
    # y = np.array([0,1,1,1])
    # # AND
    # X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # y = np.array([0, 0, 0, 1])
    # # XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    # print("np.shape(X)[1]: ", np.shape(X)[1],)
    perceptron = Perceptron(np.shape(X)[1], 0.1, 10)
    perceptron.fit(X, y)
