import copy
import numpy as np
import math

X = np.array([[1, 1], [2, 1], [0, 10], [10, 10], [5, 5], [3, 10], [9, 4], [6, 2], [2, 2], [8, 7]])
Y = np.array([[1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1]])

x_train = np.array([[1, 5], [2, 6], [2, 7], [3, 7], [3, 8], [4, 8], [5, 1], [5, 9], [6, 2], [7, 2], [7, 3], [8, 3], [8, 4], [9, 5]])
y_train = np.array([[-1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [1]])

class point:
  def __init__(self, feature, label, euclid, classify):
    self.feature = copy.copy(feature)
    self.label = copy.copy(label)
    self.classify = copy.copy(classify)
    self.euclid = copy.copy(euclid)

  def debug(self):
    print("Feature: ", self.feature, end= ' ')
    print("Label: ", self.label, end= ' ')
    print("Classify: ", self.classify, end= ' ')
    print("Euclid: ", self.euclid)

  def getEuclid(self):
    return self.euclid

def euclidian_distance(x, y):
  return ((y - x)**2)


def KNN_test(X_train, Y_train, X_test, Y_test, K):
  #print(K)
  test_points = []
  prediction = []

  for z in range(X_test.shape[0]):
    neighbors = []
    for x in range(X_train.shape[0]):
      euclid = 0
      for y in range(X_train.shape[1]):
        euclid = euclid + euclidian_distance(X_train[x][y], X_test[z][y])
      euclid = math.sqrt(euclid)
      neighbors.append(copy.copy(point(x, Y_train[x], euclid, None)))
      neighbors.sort(key=point.getEuclid)
    test_points.append(neighbors)


  for a in test_points:
    temp_predict = 0
    for itr in range(K):
      temp_predict = temp_predict + a[itr].label
    prediction.append(copy.copy(temp_predict))

  #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
  num_correct = 0
  for inc in range(X_test.shape[0]):
    if(prediction[inc] < 0 and  Y_test[inc] < 0):
      num_correct = num_correct + 1
    elif(prediction[inc] > 0 and Y_test[inc] > 0):
      num_correct = num_correct + 1

  return(num_correct/ X_test.shape[0] * 100)



def choose_K(X_train, Y_train, X_val, Y_val):

  best_training = (-1, -1)
  for i in range(Y_train.shape[0]):
    accuracy = KNN_test(X_train, Y_train, X_val, Y_val, i + 1)
    if(accuracy > best_training[0]):
      best_training = (accuracy, i)

  #print(best_training)
  return(best_training[1])

#choose_K(x_train, y_train, X, Y)
#print(KNN_test(x_train, y_train, X, Y, 1))

