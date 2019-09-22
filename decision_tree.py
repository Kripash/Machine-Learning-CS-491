import numpy as np
import math
import copy

X = np.array([[0,1,0,1],[1,1,1,1],[0,0,0,1]])
Y = np.array([[1], [0], [0]])
max_depth = 2

class Node():
  def __init__(self, value, node_left, node_right, feature):
    self.node_left = copy.copy(node_left)
    self.node_right = copy.copy(node_right)
    self.value = copy.copy(value)
    self.feature = feature

  def set_left(self, left):
    self.node_left = copy.copy(left)

  def set_right(self, right):
    self.node_right = (right)

  def set_value(self, value):
    self.value = copy.copy(value)

  def debug(self):
    if(self.node_left != None):
      self.node_left.debug()
    print("node value: " , self.value)
    print("node feature: ", self.feature)
    if(self.node_right != None):
      self.node_right.debug()

  def traverse_left(self):
    if(self.node_left != None):
      return self.node_left.traverse_left()
    else:
      return self

  def traverse_right(self):
    if (self.node_right != None):
      return self.node_right.traverse_right()
    else:
      return self


class Tree():
  def __init__(self, max_depth):
    self.root = Node(None, None, None, None)
    self.max_depth = copy.copy(max_depth)

  def set_root(self, root):
    self.root = copy.copy(root)

  def debug(self):
    self.root.debug()


#X: list of training feature data 2D numpy array
#Y: list of labels data 1D numpy array
#max_depth is the max depth for the resulting tree
def DT_train_binary(X,Y, max_depth):
  feats = []
  for features in range(X.shape[1]):
    feats.append(0)
  features_list = np.array(feats)
  #print(features_list)

  print("Features: \n", X, "\n")
  print("Labels: \n", Y, "\n")
  entropy_start = entropy_tree(Y)
  #print (entropy_start)
  root = find_root(X, Y, entropy_start)
  #print(root)
  root_node = Node(root[1], None, None, root[2])
  DT_binary_tree = Tree(max_depth)
  DT_binary_tree.set_root(root_node)
  DT_binary_tree.root.debug()
  if(DT_binary_tree.root.value == 0):
    print ("Done")
    return DT_binary_tree
  else:
    features_list[DT_binary_tree.root.feature] = 1
    entropy_subtree(X, Y, max_depth - 1, copy.copy(DT_binary_tree), copy.copy(features_list))
    #DT_binary_tree.debug()

def entropy_tree(tree):
  num_false = 0
  num_true = 0
  total_labels = tree.shape[0]
  for x in range(tree.shape[0]):
    if(tree[x][0] == 1):
      num_true = num_true + 1
    elif(tree[x][0] == 0):
      num_false = num_false + 1
  return(calc_entry(num_false, num_true, total_labels))

def entropy_subtree(features, labels, max_depth, DT_tree, features_list):
  if(max_depth == -1):
    #return entropy_subtree(features, labels, max_depth)
    return 0
  elif(max_depth == 0):
    return
  elif(len(features)== 0):
    return
  else:
    #print(features_list)
    for x in range(features.shape[1]):
      local_tree = copy.copy(DT_tree)
      if(features_list[x] == 0):
        features_list[x] = 1
        print(features_list, x)
        sub_node = Node(-1, None, None, x)
        left_sub = DT_tree.root.traverse_left()
        left_sub.set_left(sub_node)
        right_sub = DT_tree.root.traverse_right()
        right_sub.set_right(sub_node)
        #for z in range(features_list.shape[0]):
        #  if(x != z ):
        #    print (z)
        #    sub_node.set_left(z)
            #DT_tree.debug()
    #DT_tree.debug()
    #return entropy_subtree(features, labels, max_depth - 1, copy.copy(prev_entropy), copy.copy(features_list))

def find_root(features, labels, tree_entropy):
  max_entropy = [ float(-math.inf),-1, -1]
  #print(max_entropy)
  for x in range(features.shape[1]):
    #print(x)
    num_00 = 0
    num_01 = 0
    num_10 = 0
    num_11 = 0
    for y in range(labels.shape[0]):
      if(features[y][x] == 0 and labels[y] == 0):
        num_00 = num_00 + 1
      if(features[y][x] == 0 and labels[y] == 1):
        num_01 = num_01 + 1

      if(features[y][x] == 1 and labels[y] == 0):
        num_10 = num_10 + 1
      if(features[y][x] == 1 and labels[y] == 1):
        num_11 = num_11 + 1

    n_entropy = 0
    y_entropy = 0
    if(num_00 + num_01 > 0):
      n_entropy = calc_entry(num_00, num_01, (num_00 + num_01))
    #print(n_entropy, end=' ')
    if(num_10 + num_11 > 0):
      y_entropy = calc_entry(num_10, num_11, (num_10 + num_11))
    #print(y_entropy, end=' ')

    h_node = ( (((num_00 + num_01)/(labels.shape[0])) * n_entropy) +
                        (((num_10 + num_11)/(labels.shape[0])) * y_entropy) )
    #print(h_node, end = ' ')
    IG = tree_entropy - h_node
    if(IG > max_entropy[0]):
      max_entropy = (IG, h_node, x)

  return max_entropy

def calc_entry(n, y, total):
  if(n == 0 and y > 0):
    return - 0 - (((y/total) * math.log(y/total, 2)))
  elif(y == 0 and n > 0):
    return -((n / total) * math.log(n/total, 2))- 0
  else:
    return -((n / total) * math.log(n/total, 2)) - ((y/total) * math.log(y/total, 2))


def DT_test_binary(X,Y,DT):
  print("test")


def DT_train_binary_best(X_train, Y_train, X_val, Y_val):
  print("best")


DT_train_binary(X, Y, max_depth)