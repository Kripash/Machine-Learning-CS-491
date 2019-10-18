import copy
import numpy as np
import math


"""
KNN point object:
feature: feature index of the object
label: label of the point
classify: classification of the point (for prediction)
euclid: euclid distance of point to current test
"""
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


"""
euclidian_distance(x,y):
  calculate the euclidian using (y - x)^2 
  we square root the entire sum later on
"""
def euclidian_distance(x, y):
  return ((y - x)**2)

"""
KNN_test(X_train, Y_train, X_test, Y_test, K):
  The initial loop will:
  iterate through the number of testing data
  For each testing data, iterate through the training data 
  and for each feature, calculate the euclidian distance 
  
  After that has been done, each testing data will then have the 
  data sorted in ascending order based on the euclidian distance and append it to test points 
  
  The function will loop through the test points and based on the K value, it will go from the ordered list 
  based on euclidian distance up till the Kth element and calculate the prediciton for that test label 
  and append it to the prediction list. 
  
  After the prediction step is complete, the function will go through the number of x_test and check to see 
  if the sign of the prediction matches the sign of the label for that data set and calculate the number of correct
  predictions. The function will then return the accuracy as a prediction.
"""
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

  return(num_correct/ X_test.shape[0])



"""
choose_K(X_train, Y_train, X_val, Y_val):
  Set a tuple to be acurracy of -1, and K value of -1 
  go in a for loop within the range of the Y_training data set
  call KNN_test and increment K in the range of Y_training data set 
  if the accurracy is greater than the current stored in the tuple, 
  store that accuracy as tuple object 1 and the K as tuple object 2
  return the tuple object 2 for k value
"""
def choose_K(X_train, Y_train, X_val, Y_val):

  best_training = (-1, -1)
  for i in range(Y_train.shape[0]):
    accuracy = KNN_test(X_train, Y_train, X_val, Y_val, i + 1)
    if(accuracy > best_training[0]):
      best_training = (accuracy, i + 1)

  #print(best_training)
  return(best_training[1])


