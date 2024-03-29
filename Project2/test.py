#Test all necessary functions in here

import nearest_neighbors as nn
import perceptron as p
import clustering as c

import numpy as np

X = np.array([[1, 1], [2, 1], [0, 10], [10, 10], [5, 5], [3, 10], [9, 4], [6, 2], [2, 2], [8, 7]])
Y = np.array([[1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1]])

x_train = np.array([[1, 5], [2, 6], [2, 7], [3, 7], [3, 8], [4, 8], [5, 1], [5, 9], [6, 2], [7, 2], [7, 3], [8, 3], [8, 4], [9, 5]])
y_train = np.array([[-1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [1]])

print(nn.choose_K(x_train, y_train, X, Y))
print(nn.KNN_test(x_train, y_train, X, Y, 9))



A = np.array([[0,1] , [1,0] , [5,4] , [1,1] , [3,3] , [2,4] , [1,6] ])
B = np.array([[1], [1], [0], [1], [0], [0], [0]])

C = np.array([[-2,1] , [1,1] , [1.5,-0.5] , [-2,-1] , [-1,-1.5] , [2,-2] ])
D = np.array([[1], [1], [1], [-1], [-1], [-1]])

W = p.perceptron_train(C, D)
print(W[0], W[1])

accuracy = p.perceptron_test(C, D, W[0], W[1])

print(accuracy)
