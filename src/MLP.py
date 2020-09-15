import numpy as np
import os.path

from src.Layer import Layer


class MLP:
    """
    Represents a neural network.
    """
    def __init__(self):
        self._layers = []

    """
    Adds a layer to the neural network
    :param Layer layer: The layer to add
    """
    def add_layer(self, layer):
        self._layers.append(layer)

    """
    Feeds the output of the previous layer to the next layer for each layer in the MLP
    :param X: the input, 
    """
    def feed_forward(self, X):

        for layer in self._layers:
            X = layer.activate(X)

        return X

    """
    Predicts a class (probably not the right way to do this but not sure)
    :param X: input
    :return: The prediction class
    """
    def predict(self, X):
        output = self.feed_forward(X)
        if output.ndim == 1:
            return np.argmax(output) + 1

        return np.argmax(output, axis=1) + 1

    """
    Performs the back propagation algorithm and updates the weights of the neural network
    :param X: the input
    :param y: the target classes
    :param lr: the learning rate of the perceptron
    """
    def back_propagation(self, X, y, lr):
        output = self.feed_forward(X)

        # Loop over the layers backward
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]

            # If this is the output layer
            if layer == self._layers[-1]:
                target_value = np.zeros(len(output),)
                target_value[y-1] = 1
                layer.error = target_value - output
                layer.delta = layer.error * layer.error_gradient(output)
            else:
                next_layer = self._layers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.error_gradient(layer.last_activation)

        # Update the weights
        for i in range(len(self._layers)):
            layer = self._layers[i]
            # The input is either the previous layers output or X itself (for the first hidden layer)
            input_to_use = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)
            layer.weights += layer.delta * input_to_use.T * lr

        return output

    """
    Trains the neural network
    :param X: input
    :param y: the target values
    :param lr: the learning rate
    :param iterations: number of iterations the mlp needs to run
    :return: list of calculated mean square errors(we may have to use sum of squared error instead)
    """
    def train(self, features_t, targets_t, features_v, targets_v, learning_rate, iterations):
        square_errors = []
        validation = 0.0
        iteration = 0
        validations = []

        means = [0]
        for i in range(iterations):
            validations = []
            for k in range(len(features_t)):
                for j in range(len(features_t[0])):
                    output = self.back_propagation(features_t[k][j], targets_t[k][j], learning_rate)

                    target_value = np.zeros(len(output),)
                    target_value[targets_t[k][j]-1] = 1
                    mse = np.mean(np.square(target_value - output))
                    square_errors.append(mse)

                new_validation = self.compare_prediction(features_v[k], targets_v[k])
                validations.append(new_validation)

            mean = 0
            for l in validations:
                mean += l
            mean = mean/len(validations)
            print("mean after iteration: "+str(i)+" mean: "+str(mean))
            if mean > means[-1]:
                means.append(mean)
                iteration += 1
            else:
                break

        return validations, square_errors
    """
    Creates a 2d matrix of the shape [n_samples][n_features] ([7854][10] in our case)
    """
    def load_features(self):
        features_file = "./../data/features.txt"
        f = open(features_file)
        data = []

        for line in f:
            lines = line.split(",")
            for i in range(len(lines)):
                lines[i] = float(lines[i])

            data.append(lines)

        return data

    """
   Creates a 2d matrix of the shape [n_samples][n_features] ([7854][10] in our case)
   """
    def load_unknown(self):
        features_file = "../data/unknown.txt"
        f = open(features_file)
        data = []

        for line in f:
            lines = line.split(",")
            for i in range(len(lines)):
                lines[i] = float(lines[i])

            data.append(lines)

        return data

    "Creates a 2d matrix of the outputs for each input. Shape[n_samples][target_class] ([7854][1] in our case)"
    def load_targets(self):
        targets_file = "./../data/targets.txt"
        f = open(targets_file)
        targets = []
        for target in f:
            targets.append(target)

        for i in range(0, len(targets)):
            targets[i] = int(targets[i])

        targets = np.reshape(targets, (-1, 1))

        return targets

    """
    Compares MLP results of classifying features to the target results.
    feat - features to predict the result from
    results - target values of the features
    return - value between 0 and 1 that represents how much % of the features got properly matched
    """
    def compare_prediction(self, feat, results):
        prediction = self.predict(feat)
        total = 0
        for i in range(len(prediction)):
            if results[i] == prediction[i]:
                total += 1
        correct = total/len(prediction)
        return correct

    def compare_prediction_confusion_matrix(self, feat, results):
        # horizontal - actual class
        # vertical - predicted class
        outputs = []

        confusion = np.zeros((7, 7))
        for i in range(len(feat)):
            output = self.feed_forward(feat[i])
            result = 0
            if output.ndim == 1:
                result = np.argmax(output)
            else:
                result = np.argmax(output, axis=1)
            print(result)
            outputs.append(result)
            confusion[result, results[i][0]-1] = confusion[result, results[i][0]-1]+1

        self.writeToFile("../data/Group_28_classes.txt", outputs)

        return confusion

    def writeToFile(self, fileName, input):
        file = fileName
        f = open(file, "a+")
        data = []
        f.write(str(input[0]))
        for i in input[1:]:
            f.write(", " + str(i))

        f.close()
        return data


if __name__ == '__main__':
    nn = MLP()
    nn.add_layer(Layer(10, 15))
    nn.add_layer(Layer(15, 15))
    nn.add_layer(Layer(15, 7))

    features = nn.load_features()
    targets = nn.load_targets()

    print(np.shape(features))

    features_train = [features[1000:7000],
                      np.append(features[:1000], features[2000:7000], axis=0),
                      np.append(features[:2000], features[3000:7000], axis=0),
                      np.append(features[:3000], features[4000:7000], axis=0),
                      np.append(features[:4000], features[5000:7000], axis=0),
                      np.append(features[:5000], features[6000:7000], axis=0), features[0:6000]]

    features_validation = [features[0:1000], features[1000:2000], features[2000:3000], features[3000:4000],
                           features[4000:5000], features[5000:6000], features[6000:7000]]

    targets_train = [targets[1000:7000],
                     np.append(targets[:1000], targets[2000:7000], axis=0),
                     np.append(targets[:2000], targets[3000:7000], axis=0),
                     np.append(targets[:3000], targets[4000:7000], axis=0),
                     np.append(targets[:4000], targets[5000:7000], axis=0),
                     np.append(targets[:5000], targets[6000:7000], axis=0), targets[0:6000]]
    targets_validation = [targets[0:1000], targets[1000:2000], targets[2000:3000], targets[3000:4000],
                          targets[4000:5000], targets[5000:6000], targets[6000:7000]]

    features_test = features[7000:]
    targets_test = targets[7000:]

    features_unknown = nn.load_unknown()

    # Train the neural network
    validations, errors = nn.train(features_train, targets_train, features_validation, targets_validation, 0.02, 1000)

    predict = nn.compare_prediction(features_test, targets_test)
    confusion = nn.compare_prediction_confusion_matrix(features_test, targets_test)
    predict2 = nn.predict(features_unknown)

    print("percentage of correct predictions: "+str(predict))
    print(confusion)