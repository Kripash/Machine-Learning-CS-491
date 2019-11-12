import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Apply softmax to output-classification array.
    :param z: (1 x num_classes) output-classification.
    :return: (1 x num_classes) output-classification with softmax applied.
    """
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def predict(model: Dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    """
    Predict the label of a sample input
    :param model: neural network model for prediction
    :param x: (1 x inputs) sample input
    :return: predicted label
    """
    # Calculate hidden layer
    a: np.ndarray = np.dot(x, model['W1']) + model['b1']
    # Pass hidden layer through tanh function
    h: np.ndarray = np.tanh(a)
    # Calculate output
    z: np.ndarray = np.dot(h, model['W2']) + model['b2']
    # Return index of softmax array that has the highest classification value
    return np.argmax(softmax(z), axis=1)


def calculate_loss(model: Dict[str, np.ndarray], X: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate training loss
    :param model: neural network model for prediction
    :param X: (samples x inputs) array of samples' inputs
    :param y: (labels) array of labels
    :return: loss
    """
    # Convert labels to classification vectors
    y: np.ndarray = np.array([[0, 1] if label else [1, 0] for label in y])

    loss: float = 0
    total_samples: int = X.shape[0]

    # for each sample
    for sample, y_truth in zip(X, y):
        # Calculate hidden layer
        a: np.ndarray = np.dot(sample, model['W1']) + model['b1']
        # Pass hidden layer through tanh function
        h: np.ndarray = np.tanh(a)
        # Calculate output classification
        z: np.ndarray = np.dot(h, model['W2']) + model['b2']
        # Apply softmax to output classification
        y_predict: np.ndarray = softmax(z)
        # Calculate loss of that classification
        loss += np.sum(y_truth * np.log(y_predict))

    return -loss / total_samples


def build_model(X: np.ndarray,
                y: np.ndarray,
                nn_hdim: int,
                num_passes: int = 20000,
                print_loss: bool = False,
                learning_rate: float = 0.01) -> Dict[str, np.ndarray]:
    """
    Train neural network model
    :param X: (samples x inputs) array of training samples' inputs
    :param y: (labels) array of training labels
    :param nn_hdim: number of hidden nodes of a hidden layer
    :param num_passes: number of epochs
    :param print_loss: print training loss to monitor
    :param learning_rate: learning rate
    :return: dictionary of model parameters
    """
    # Convert labels to classification vectors in order to gradient descent
    y_class: np.ndarray = np.array([[0, 1] if label else [1, 0] for label in y])
    # Randomly initializing weights and biases
    np.random.seed(0)
    weight_1: np.ndarray = np.random.rand(2, nn_hdim)
    bias_1: np.ndarray = np.random.rand(nn_hdim)
    weight_2: np.ndarray = np.random.rand(nn_hdim, 2)
    bias_2: np.ndarray = np.random.rand(2)
    # Initialize the model
    model: Dict[str, np.ndarray] = {'W1': weight_1, 'b1': bias_1, 'W2': weight_2, 'b2': bias_2}

    # For each epoch
    for i in range(num_passes):
        # Calculate hidden layer
        a = X.dot(weight_1) + bias_1
        # Pass hidden layer through tanh function
        h = np.tanh(a)
        # Calculate output classification
        z = h.dot(weight_2) + bias_2
        # Apply softmax to output classification
        y_hat = softmax(z)

        # Calculate gradients
        dLdy = y_hat - y_class
        dLda = (1 - np.power(h, 2)) * dLdy.dot(weight_2.T)
        dLw1 = np.dot(X.T, dLda)
        dLb1 = np.sum(dLda, axis=0, keepdims=True)
        dLw2 = h.T.dot(dLdy)
        dLb2 = np.sum(dLdy, axis=0, keepdims=True)

        # Update weights and biases
        weight_1 = weight_1 - (learning_rate * dLw1)
        bias_1 = bias_1 - (learning_rate * dLb1)
        weight_2 = weight_2 - (learning_rate * dLw2)
        bias_2 = bias_2 - (learning_rate * dLb2)

        # Update model with new weights and biases
        model = {'W1': weight_1, 'b1': bias_1, 'W2': weight_2, 'b2': bias_2}

        # Print training loss to monitor
        if print_loss and i % 1000 == 0:
            print(calculate_loss(model, X, y))

    return model


def plot_decision_boundary(pred_func, X, y):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
