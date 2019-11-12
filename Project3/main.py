import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from math import log
from typing import Dict
from neural_network import *

if __name__ == '__main__':
    np.random.seed(0)
    X, y = make_moons(200, noise=0.2)
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.figure(figsize=(16, 32))
    hidden_layer_dimensions = [1, 2, 4, 5]
    for i, nn_hdim in enumerate(hidden_layer_dimensions):
        plt.subplot(5, 2, i + 1)
        plt.title('HiddenLayerSize%d' % nn_hdim)
        model = build_model(X, y, nn_hdim, print_loss=True)
        plot_decision_boundary(lambda x: predict(model, x), X, y)
    plt.show()
