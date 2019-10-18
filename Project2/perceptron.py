import numpy as np
import math
import copy

#X = np.array([[0,1] , [1,0] , [5,4] , [1,1] , [3,3] , [2,4] , [1,6] ])
#Y = np.array([[1], [1], [0], [1], [0], [0], [0]])

#X = np.array([[-2,1] , [1,1] , [1.5,-0.5] , [-2,-1] , [-1,-1.5] , [2,-2] ])
#Y = np.array([[1], [1], [1], [-1], [-1], [-1]])

"""
Perceptron object:
bias: bias of the perceptron 
weights: weights of the perceptron 
data: the features of data set
labels: labels of data set
epoch_variable: number of epochs
last_weight_change: track which was the last weight change
"""
class perceptron():
  def __init__(self, X, Y):
    self.bias = 0
    self.weights = np.zeros(X.shape[1])
    self.data = copy.copy(X)
    self.labels = copy.copy(Y)
    self.epoch_variable = -1
    self.last_weight_change = -1

  def debug(self):
    #print(self.data)
    #print(self.labels)
    print("weights:" , self.weights)
    print("bias: ", self.bias)
    print("Epoch", self.epoch_variable)
    print("last weight change: ", self.last_weight_change)
    print("")


"""
perceptron_train:
  The function takes in some data set X with features and some Y with labels. 
  The function goes in an infinite loop. 
  In the loop:
  The function will keep track of the last_weight_change and if it does 1 complete loop, the function will 
  return, because it has completed 1 full epoch without changing weights. 
  The function will perform the perceptron algorith. For this, we assume a data set of 0 is also -1 since 
  that is what the original data set consisted of. The function will calculate a, such that 
  a = W * X + bias 
  if a * Y <= 0: 
  update perceptron so that:
  W = W + X * Y 
  bias = bias + Y
"""
def perceptron_train(X, Y):
  perceptron_local = perceptron(X, Y)
  #perceptron_local.debug()

  while(1):
    for samples in range(X.shape[0]):

      if(perceptron_local.last_weight_change == samples):
        #print("STOP")
        return (perceptron_local.weights, perceptron_local.bias)

      if(samples == 0):
        perceptron_local.epoch_variable = perceptron_local.epoch_variable + 1

      a = np.dot(perceptron_local.weights, X[samples]) + perceptron_local.bias
      update = None

      if(Y[samples] == 0):
        update = a * -1
      else:
        update = a * Y[samples]

      if(update <= 0):
        perceptron_local.last_weight_change = samples

        if(Y[samples] == 0):
          perceptron_local.weights = copy.copy(perceptron_local.weights) + (X[samples] * -1)
          perceptron_local.bias = perceptron_local.bias + -1
        else:
          perceptron_local.weights = copy.copy(perceptron_local.weights) + ((X[samples] * Y[samples]))
          perceptron_local.bias = perceptron_local.bias + Y[samples]


      #print("sample: ", samples)
      #print("update: ", update)
      #perceptron_local.debug()

    #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

"""
perceptron_test:
  The function will take in some X features of test set and Y labels of test data and the weights and bias. 
  The function will go through the number of labels and calculate the prediction using: 
  prediction = W * X + b
  If the sign of the prediction is equal to the label, the function increments the number of correct predictions by 1. 
  The function will return the accuracy based on num of correct predictions over the total test set.
"""

def perceptron_test(X_test, Y_test, w, b):
  num_correct = 0
  for i in range(X_test.shape[0]):
    if(Y_test[i] == 0):
      Y_test[i] = -1
    label = np.dot(w, X_test[i]) + b
    #print(w, " ", X_test[i], " ", b)
    #print(label, " ", Y_test[i])
    if(label < 0 and Y_test[i] < 0):
        num_correct = num_correct + 1
    elif(label >= 0 and Y_test[i] >= 0):
        num_correct = num_correct + 1

  return num_correct/X_test.shape[0]



#W = perceptron_train(X, Y)

#accuracy = perceptron_test(X, Y, W[0], W[1])

#print(accuracy)
