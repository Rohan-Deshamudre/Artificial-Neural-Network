import numpy as np


class Layer:
    """
    Represents a layer (hidden or output) in our neural network.
    :param n_input: input size (number of features)
    :param n_neurons: number of neurons in this layer
    :param weights: the layers weights
    :param bias: the layers bias
    """
    def __init__(self, n_input, n_neurons, weights=None, bias=None):
        # self.weights = weights if weights is not None else np.full((n_input, n_neurons), 1.0)
        self.weights = weights if weights is not None else np.random.uniform(-2.4/n_input, 2.4/n_input, size=(n_input,n_neurons))
        self.bias = bias if bias is not None else np.random.uniform(-2.4/n_input, 2.4/n_input, n_neurons)
        self.last_activation = None
        self.error = None
        self.delta = None


    """
    Calculates the dot product of the weights and the inputs and applies the sigmoid activation function to it
    :param X: the input. which I guess is a vector of the size of input
    :return: the result which is a float I guess? 
    """
    def activate(self, x):

        net_res = np.dot(x, self.weights) - self.bias
        self.last_activation = 1 / (1 + np.exp(-net_res))
        return self.last_activation

    """
    Applies the derivative of the sigma. (check slide 35 and 36 in the Chapter6:Neural Networks, supervised ppt)
    :param r: The normal value.
    :return: The "derived" value.
    """
    def error_gradient(self, pred):
        return pred * (1 - pred)

