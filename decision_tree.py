import numpy as np
import math

X = np.array([[0,1,0,1],[1,1,1,1],[0,0,0,1]])
Y = np.array([[1], [1], [0]])
max_depth = 2

class Node():
  def __init__(self, value, node_left, node_right):
    self.node_left = node_left
    self.node_right = node_right
    self.value = value

  def set_left(self, left):
    self.node_left = left

  def set_right(self, right):
    self.node_right = right

  def set_value(self, value):
    self.value = value

  def debug(self):
    if(self.node_left != None):
      self.node_left.debug()
    print(self.value)
    if(self.node_right != None):
      self.node_right.debug()




class Tree():
  def __init__(self, max_depth):
    self.root = Node(None, None, None)
    self.max_depth = max_depth

  def set_root(self, root):
    self.root = root

  def debug(self):
    self.root.debug()


#X: list of training feature data 2D numpy array
#Y: list of labels data 1D numpy array
#max_depth is the max depth for the resulting tree
def DT_train_binary(X,Y, max_depth):
  print("Features: \n", X)
  print("Labels: \n", Y)
  print(Y.shape)
  DT_tree = Tree(max_depth)
  DT_tree.debug()


def DT_test_binary(X,Y,DT):
  print("test")


def DT_train_binary_best(X_train, Y_train, X_val, Y_val):
  print("best")


DT_train_binary(X, Y, max_depth)