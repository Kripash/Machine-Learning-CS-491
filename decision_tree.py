import numpy as np
import math
import copy

X = np.array([[0,1,0,1],[1,1,1,1],[0,0,0,1]])
Y = np.array([[1], [1], [0]])
max_depth = 2

feature_dict = {}

class Node():
  def __init__(self, value, node_left, node_right):
    self.node_left = copy.copy(node_left)
    self.node_right = copy.copy(node_right)
    self.value = copy.copy(value)

  def set_left(self, left):
    self.node_left = copy.copy(left)

  def set_right(self, right):
    self.node_right = copy.copy(right)

  def set_value(self, value):
    self.value = copy.copy(value)

  def debug(self):
    if(self.node_left != None):
      self.node_left.debug()
    print(self.value)
    if(self.node_right != None):
      self.node_right.debug()


class Tree():
  def __init__(self, max_depth):
    self.root = Node(None, None, None)
    self.max_depth = copy.copy(max_depth)

  def set_root(self, root):
    self.root = copy.copy(root)

  def debug(self):
    self.root.debug()


#X: list of training feature data 2D numpy array
#Y: list of labels data 1D numpy array
#max_depth is the max depth for the resulting tree
def DT_train_binary(X,Y, max_depth):
  for features in range(X.shape[1]):
    feature_dict.update({str(features): 0})

  for x , y in feature_dict.items():
    print(x, y)

  print("Features: \n", X, "\n")
  print("Labels: \n", Y, "\n")
  #samples = X.shape[0]
  #features = X.shape[1]
  #labels = Y.shape[0]
  print(entropy_tree(Y))
  entropy_subtree(X, Y, max_depth)
  #DT_tree = Tree(max_depth)
  #DT_tree.debug()

def entropy_tree(tree):
  num_false = 0
  num_true = 0
  total_features = tree.shape[0]
  for x in range(tree.shape[0]):
    #print(tree[x][0])
    if(tree[x][0] == 1):
      num_true = num_true + 1
    elif(tree[x][0] == 0):
      num_false = num_false + 1
  #print("num_true: ", num_true)
  #print("num_false: ", num_false)
  return(calc_entry(num_false, num_true, total_features))


def entropy_subtree(features, labels, max_depth):
  if(max_depth == -1):
    return entropy_subtree(features, labels, max_depth)
  elif(max_depth == 0):
    return
  else:
    print(max_depth)

    return entropy_subtree(features, labels, max_depth - 1)


def calc_entry(n, y, total):
  return ((-n / total) * math.log(n/total, 2)) - ((y/total) * math.log(y/total, 2))


def DT_test_binary(X,Y,DT):
  print("test")


def DT_train_binary_best(X_train, Y_train, X_val, Y_val):
  print("best")


DT_train_binary(X, Y, max_depth)