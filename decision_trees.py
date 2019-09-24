import sys
import numpy as np
import math
import copy

class Node():
  def __init__(self, value, node_left, node_right, feature, left, right):
    self.node_left = copy.copy(node_left)
    self.node_right = copy.copy(node_right)
    self.h_value = copy.copy(value)
    self.h_left = None
    self.h_right = None
    self.feature = copy.copy(feature)
    self.L_value = copy.copy(left)
    self.R_value = copy.copy(right)
    self.path = []

  def set_left(self, left):
    self.node_left = left

  def set_right(self, right):
    self.node_right = right

  def set_value(self, value):
    self.value = copy.copy(value)

  def copy_path(self, path):
    self.path = copy.copy(path)

  def append_path(self, path):
    self.path.append(copy.copy(path))

  def debug(self):
    if(self.node_left != None):
      self.node_left.debug()
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("node h_value: " , self.h_value)
    print("node h_left: ", self.h_left)
    print("node h_right: ", self.h_right)
    print("node feature: ", self.feature)
    print("node L_value: ", self.L_value)
    print("node R_value: ", self.R_value)
    print("left node: " , self.node_left)
    print("right node",self.node_right)
    print("path: \n", self.path)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
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
    self.root = Node(None, None, None, None, None, None)
    self.max_depth = copy.copy(max_depth)
    self.label = -1

  def set_root(self, root):
    self.root = root

  def debug(self):
    print("Printing Tree data in order!(left, self, right) recursively")
    self.root.debug()
    print("Finishing Printing Tree!")


#X: list of training feature data 2D numpy array
#Y: list of labels data 1D numpy array
#max_depth is the max depth for the resulting tree
def DT_train_binary(X,Y, max_depth):
  feats = []
  for features in range(X.shape[1]):
    feats.append(0)
  features_list = np.array(feats)
  #print(features_list)

  #print("Features: \n", X, "\n")
  #print("Labels: \n", Y, "\n")
  entropy_start = entropy_tree(Y)
  if(max_depth == 0):
    initial_label = find_root(X, Y, entropy_start, max_depth)
    DT_binary_tree = Tree(max_depth)
    DT_binary_tree.label = initial_label
    return DT_binary_tree

  root = find_root(X, Y, entropy_start, max_depth)
  #print(root)
  #print(root)
  root_node = Node(root[1], None, None, root[2], root[3], root[4])
  root_node.h_left = root_node.h_value
  root_node.h_right = root_node.h_value
  #root_node.append_path((0,0))
  DT_binary_tree = Tree(max_depth)
  DT_binary_tree.set_root(root_node)
  if(DT_binary_tree.root.h_value == 0):
    #print ("Done")
    #DT_binary_tree.debug()
    return DT_binary_tree
  else:
    features_list[DT_binary_tree.root.feature] = 1
    entropy_subtree(X, Y, max_depth - 1, DT_binary_tree, root_node, copy.copy(features_list))
    #DT_binary_tree.debug()
    return DT_binary_tree

def entropy_tree(tree):
  num_false = 0
  num_true = 0
  total_labels = tree.shape[0]
  for x in range(tree.shape[0]):
    if(tree[x][0] == 1):
      num_true = num_true + 1
    elif(tree[x][0] == 0):
      num_false = num_false + 1
  return(calc_entropy(num_false, num_true, total_labels))

def entropy_subtree(features, labels, max_depth, DT_tree, curr_node, features_list):
  if(max_depth <= -1):
    feat_count = 0
    for x in features_list:
      if x == 0:
        feat_count = feat_count + 1
    if feat_count == 0:
      return
  elif(max_depth == 0):
    return

  """" Otherwise We need to split somewhere if possible """
  max_entropy = [ float(-math.inf),-1, -1, -1, -1, -1]
  right_entropy = [ float(-math.inf),-1, -1, -1, -1, -1]

  n_entropy = 0
  y_entropy = 0
  rn_entropy = 0
  ry_entropy = 0

  for x in range(features.shape[1]):
    local_tree = copy.copy(DT_tree)
    if(features_list[x] == 0):
      """ THIS IS ONLY FOR THE LEFT SIDE """
      sub_node = Node(-1, None, None, x, None, None)
      sub_node.copy_path(curr_node.path)
      sub_node.append_path((curr_node.feature, 0))
      num_00 = 0
      num_01 = 0
      num_10 = 0
      num_11 = 0
      cross_index = []
      for y in range(labels.shape[0]):
        cross_index.append(y)
      cross_index_copy = copy.copy(cross_index)
      for i in range(len(sub_node.path)):
        feature_index = sub_node.path[i][0]
        feature_val = sub_node.path[i][1]
        for y in cross_index:
          if(features[y][feature_index] != feature_val):
            try:
              cross_index_copy.remove(y)
            except:
              pass
              #print("left", end = ' ')
              #print(y, cross_index_copy)
      cross_index = copy.copy(cross_index_copy)
      #print(cross_index)
      for y in cross_index:
        #print(y,x, features[y][x])
        if (features[y][x] == 0 and labels[y] == 0):
          num_00 = num_00 + 1
        if (features[y][x] == 0 and labels[y] == 1):
          num_01 = num_01 + 1
        if (features[y][x] == 1 and labels[y] == 0):
          num_10 = num_10 + 1
        if (features[y][x] == 1 and labels[y] == 1):
          num_11 = num_11 + 1

      if (num_00 + num_01 > 0):
        n_entropy = calc_entropy(num_00, num_01, (num_00 + num_01))
      if (num_10 + num_11 > 0):
        y_entropy = calc_entropy(num_10, num_11, (num_10 + num_11))
      #print(num_00, num_01, num_10, num_11, n_entropy, y_entropy, len(cross_index))
      if(len(cross_index) > 1):
        h_node = ((((num_00 + num_01) / len(cross_index)) * n_entropy) +
                  (((num_10 + num_11) / len(cross_index)) * y_entropy))

        #print(n_entropy, y_entropy, h_node)

        IG = curr_node.h_left - h_node
        #print(curr_node.h_left, "IG: ", IG)
        if(IG > max_entropy[0]):
          if (num_00 >= num_01):
            if (num_10 >= num_11):
              max_entropy = (IG, h_node, x, 0, 0, n_entropy, y_entropy)
            elif (num_11 > num_10):
              max_entropy = (IG, h_node, x, 0, 1, n_entropy, y_entropy)
          elif (num_01 > num_00):
            if (num_10 >= num_11):
              max_entropy = (IG, h_node, x, 1, 0, n_entropy, y_entropy)
            elif (num_11 > num_10):
              max_entropy = (IG, h_node, x, 1, 1, n_entropy, y_entropy)
      """ END LEFT SIDE"""
      #print("~~~~~~~~~~~~~RIGHT SIDE~~~~~~~~~~~~~~")
      """ Computer the Right side now """
      right_node = Node(-1, None, None, x, None, None)
      right_node.copy_path(curr_node.path)
      right_node.append_path((curr_node.feature, 1))
      n_00 = 0
      n_01 = 0
      n_10 = 0
      n_11 = 0
      c_index = []
      for a in range(labels.shape[0]):
        c_index.append(a)
      c_index_copy = copy.copy(c_index)
      for b in range(len(right_node.path)):
        right_feat_index = right_node.path[b][0]
        right_feat_val = right_node.path[b][1]
        for c in c_index:
          if(features[c][right_feat_index] != right_feat_val):
            try:
              c_index_copy.remove(c)
            except:
              pass
              #print("right", end = ' ')
              #print(c, cross_index_copy)
      c_index = copy.copy(c_index_copy)
      for c in c_index:
        if (features[c][x] == 0 and labels[c] == 0):
          n_00 = n_00 + 1
        if (features[c][x] == 0 and labels[c] == 1):
          n_01 = n_01 + 1
        if (features[c][x] == 1 and labels[c] == 0):
          n_10 = n_10 + 1
        if (features[c][x] == 1 and labels[c] == 1):
          n_11 = n_11 + 1

      if (n_00 + n_01 > 0):
        rn_entropy = calc_entropy(n_00, n_01, (n_00 + n_01))
      if (n_10 + n_11 > 0):
        ry_entropy = calc_entropy(n_10, n_11, (n_10 + n_11))
      if(len(c_index) > 1):
        h_right = ((((n_00 + n_01) / len(c_index)) * rn_entropy) +
                  (((n_10 + n_11) / len(c_index)) * ry_entropy))

        R_IG = curr_node.h_right - h_right
        #print(curr_node.h_right, "IG: ", IG)
        if(R_IG > right_entropy[0]):
          if (n_00 >= n_01):
            if (n_10 >= n_11):
              right_entropy = (R_IG, h_right, x, 0, 0, rn_entropy, ry_entropy)
            elif (n_11 > n_10):
              right_entropy = (R_IG, h_right, x, 0, 1, rn_entropy, ry_entropy)
          elif (n_01 > n_00):
            if (n_10 >= n_11):
              right_entropy = (R_IG, h_right, x, 1, 0, rn_entropy, ry_entropy)
            elif (num_11 > num_10):
              right_entropy = (R_IG, h_right, x, 1, 1, rn_entropy, ry_entropy)
      features_list[x] = 0
      #print("~~~~~~~~~~~~~~RIGHT SIDE FINISHED~~~~~~~")
  #print(features_list)
  #print(max_entropy)
  #print(right_entropy)

  #recursion_left
  if(max_entropy[2] != -1):
    features_left = copy.copy(features_list)
    features_left[max_entropy[2]] = 1
    #print(features_left)
    sub_node.h_value = max_entropy[1]
    sub_node.h_left = max_entropy[5]
    sub_node.h_right = max_entropy[6]
    sub_node.feature = max_entropy[2]
    sub_node.L_value = max_entropy[3]
    sub_node.R_value = max_entropy[4]
    curr_node.set_left(sub_node)
    #print(max_entropy[1])
    if(max_entropy[1] != 0):
      entropy_subtree(features, labels, max_depth -1, DT_tree, sub_node, copy.copy(features_left))

  #recursion_right
  if(right_entropy[2] != -1):
    features_right = copy.copy(features_list)
    features_right[right_entropy[2]] = 1
    #print(features_right)
    right_node.h_value = right_entropy[1]
    right_node.h_left = right_entropy[5]
    right_node.h_right = right_entropy[6]
    right_node.feature = right_entropy[2]
    right_node.L_value = right_entropy[3]
    right_node.R_value = right_entropy[4]
    curr_node.set_right(right_node)
    #print(right_entropy[1])
    if(right_entropy[1] != 0):
      entropy_subtree(features, labels, max_depth - 1, DT_tree, right_node, copy.copy(features_right))

def find_root(features, labels, tree_entropy, max_depth):

  if(max_depth == 0):
    num_0 = 0
    num_1 = 0
    for x in range(labels.shape[0]):
      if(labels[x][0] == 0):
        num_0 = num_0 + 1
      elif(labels[x][0] == 1):
        num_1 = num_1 + 1
    if(num_0 >= num_1):
      return 0
    elif(num_1 > num_0):
      return 1

  max_entropy = [ float(-math.inf),-1, -1, -1, -1]
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
      n_entropy = calc_entropy(num_00, num_01, (num_00 + num_01))
    #print(n_entropy, end=' ')
    if(num_10 + num_11 > 0):
      y_entropy = calc_entropy(num_10, num_11, (num_10 + num_11))
    #print(y_entropy, end=' ')

    h_node = ( (((num_00 + num_01)/(labels.shape[0])) * n_entropy) +
                        (((num_10 + num_11)/(labels.shape[0])) * y_entropy) )
    #print("node : ", x , h_node)
    #print((num_00 +  num_01), (num_10 + num_11), n_entropy, y_entropy)
    IG = tree_entropy - h_node
    if(IG > max_entropy[0]):
      if(num_00 >= num_01):
        if(num_10 >= num_11):
          max_entropy = (IG, h_node, x, 0, 0)
        elif(num_11 > num_10):
          max_entropy = (IG, h_node, x, 0, 1)
      elif (num_01 > num_00):
        if (num_10 >= num_11):
          max_entropy = (IG, h_node, x, 1, 0)
        elif (num_11 > num_10):
          max_entropy = (IG, h_node, x, 1, 1)
  return max_entropy

def calc_entropy(n, y, total):
  if(n == 0 and y > 0):
    return - 0 - (((y/total) * math.log(y/total, 2)))
  elif(y == 0 and n > 0):
    return -((n / total) * math.log(n/total, 2))- 0
  else:
    return -((n / total) * math.log(n/total, 2)) - ((y/total) * math.log(y/total, 2))


def DT_test_binary(X,Y,DT):
  #print(Y.shape[0])
  if(DT.max_depth == 0):
    num_correct = 0
    for labels in range(Y.shape[0]):
      if(Y[labels] == DT.label):
        num_correct = num_correct + 1
    #print((num_correct) / (Y.shape[0]))
    return (num_correct/ Y.shape[0]) * 100

  num_correct = 0
  this_node = DT.root
  for x in range(Y.shape[0]):
    this_feature = this_node.feature
    direction = X[x][this_feature]
    if(direction == 0 and this_node.node_left == None):
      if(this_node.L_value == Y[x]):
        num_correct = num_correct + 1
    elif(direction == 0 and this_node.node_left != None):
      num_correct = num_correct + DT_test_binary_helper(X[x], Y[x], this_node.node_left)
    elif(direction == 1 and this_node.node_right == None):
      if(this_node.R_value == Y[x]):
        num_correct = num_correct + 1
    elif(direction == 1 and this_node.node_right != None):
      num_correct = num_correct + DT_test_binary_helper(X[x], Y[x], this_node.node_right)

  #print((num_correct)/(Y.shape[0]))
  return(num_correct/ Y.shape[0]) * 100


def DT_test_binary_helper(sample, label, this_node):
  #print("In DT_test_binary_helper")
  this_feature = this_node.feature
  direction = sample[this_feature]
  #print(sample, label, this_feature, direction)
  num_correct = 0
  if (direction == 0 and this_node.node_left == None):
    if (this_node.L_value == label):
      num_correct = num_correct + 1
  elif (direction == 0 and this_node.node_left != None):
    num_correct = num_correct + DT_test_binary_helper(sample, label, this_node.node_left)
  elif (direction == 1 and this_node.node_right == None):
    if (this_node.R_value == label):
      num_correct = num_correct + 1
  elif (direction == 1 and this_node.node_right != None):
    num_correct = num_correct + DT_test_binary_helper(sample, label, this_node.node_right)

  return num_correct



def DT_train_binary_best(X_train, Y_train, X_val, Y_val):
  best_tree = (None, -1)
  for depth in range(X_train.shape[1]):
    tree = DT_train_binary(X_train, Y_train, depth)
    accuracy = DT_test_binary(X_val, Y_val, tree)
    #tree.debug()
    #print(accuracy, tree.max_depth)
    if(accuracy > best_tree[1]):
      best_tree = (tree, accuracy)
      #print ("best tree: " , accuracy)
  return (best_tree[0])



def DT_make_prediction(x, DT):
  if(DT.max_depth == 0):
    return DT.label
  feature = DT.root.feature
  direction = x[feature]
  if(direction == 0):
    if(DT.root.node_left == None):
      return DT.root.L_value
    elif(DT.root.node_left != None):
      return DT_make_prediction_helper(x, DT.root.node_left)
  elif(direction == 1):
    if(DT.root.node_right == None):
      return DT.root.R_value
    elif(DT.root.node_right != None):
      return DT_make_prediction_helper(x, DT.root.node_right)

def DT_make_prediction_helper(x,this_node):
  feature = this_node.feature
  direction = x[feature]
  if(direction == 0):
    if(this_node.node_left == None):
      return this_node.L_value
    elif(this_node.node_right != None):
      return DT_make_prediction_helper(x, this_node.node_left)
  elif(direction == 1):
    if(this_node.node_right == None):
      return this_node.R_value
    elif(this_node.node_right != None):
      return DT_make_prediction_helper(x, this_node.node_right)