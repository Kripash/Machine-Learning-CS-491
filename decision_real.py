#author: Kripash Shrestha
#Project 1 Machine Learning CS491 UNR
#Dr.Emily Hand

import numpy as np
import math
import copy

"""
Node object for the decision tree
Each node contains:
  a node_left and node_right (children) which are none if the node does not have children.
  a h_value for the overall h value of the node 
  a h_left and h_right value for the individual entropy values of the left and right branches 
  a feature that represents the feature that the node splits at 
  L_value and R_value which represent the values for traversing left and right respectively 
  the path that was taken to get to the node
"""
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

"""
  This represents the overall tree object for the decision tree.
  The tree contains a root for the tree and if that is none, it contains a label that represents the entire tree. 
"""
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


"""DT_train_real
   Train the binary tree based on the entropy of the entire tree and the root node split if there is one. 
   As long as there are features to split on and the IG is not 0, then call entropy_subtree to build the rest 
   of the tree. 
   Return the binary tree for all 3 cases.
   The cases are: 
   1. max depth is 0 at the start so return the most occuring label 
   2. you only have to split at the root node so return that 
   3. There are more splits that can be done so compute the splits, build the tree and return that tree.
"""
def DT_train_real(X,Y, max_depth):
  feats = []
  #set up all of the possible features
  for features in range(X.shape[1]):
    feats.append(0)
  features_list = np.array(feats)

  #get the entropy of the entire tree
  entropy_start = entropy_tree(Y)
  #if max_depth is 0, take the label that occurs the most and return the tree with that
  if(max_depth == 0):
    initial_label = find_root(X, Y, entropy_start, max_depth)
    DT_binary_tree = Tree(max_depth)
    DT_binary_tree.label = initial_label
    return DT_binary_tree

  #otherwise find the root node split
  root = find_root(X, Y, entropy_start, max_depth)
  root_node = Node(root[1], None, None, root[2], root[3], root[4])
  root_node.h_left = root_node.h_value
  root_node.h_right = root_node.h_value
  DT_binary_tree = Tree(max_depth)
  DT_binary_tree.set_root(root_node)
  #if the entropy is 0, there is nothing left to split on so we return the tree, otherwise call entropy_subtree to
  #build the tree recursively and return the tree
  if(DT_binary_tree.root.h_value == 0):
    #print ("Done")
    #DT_binary_tree.debug()
    return DT_binary_tree
  else:
    features_list[DT_binary_tree.root.feature] = 1
    entropy_subtree(X, Y, max_depth - 1, DT_binary_tree, root_node, copy.copy(features_list))
    #DT_binary_tree.debug()
    return DT_binary_tree

"""entropy_tree
    takes in parameter 'tree' which is the labels and calculates the entropy of the tree by comparing the labels with 
    the false and true values for the binary tree and calculates the entropy and returns that
"""
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

""" entropy_subtree
    The function is used to build the remainer of the tree if possible. 
    The stopping conditions are:
    1. If the max_depth is 0, the function returns as there is nothing left to split on. 
    2. If the max_depth is smaller than or equal to -1, the function has to keep in track the features left 
       to build the tree since a parameter of -1 will go until IG is 0 or there is nothing left to split on.
       
    The function will find all of the features left to split on and the samples that correspond to the features and path 
    taken for the current node. The function will calculate the left entropy, right entropy and entropy of the node and 
    calcualte the Information gain. The function will then take the max information gain and use that feature for the 
    split and then recursively go left and right for checking splits. 
"""
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

  #holds the entropy values for the left and right sub trees
  n_entropy = 0
  y_entropy = 0
  rn_entropy = 0
  ry_entropy = 0

  for x in range(features.shape[1]):
    local_tree = copy.copy(DT_tree)
    #if the feature has not been used yet, go ahead and try to split on it
    if(features_list[x] == 0):
      """ THIS IS ONLY FOR THE LEFT SIDE """
      sub_node = Node(-1, None, None, x, None, None)
      sub_node.copy_path(curr_node.path)
      #append the current path taken to the curr node and add a 0 since we will be going left
      sub_node.append_path((curr_node.feature, 0))
      num_00 = 0
      num_01 = 0
      num_10 = 0
      num_11 = 0
      cross_index = []
      #find which samples can be used for the current path and feature
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
      #for the current samples left, find the left traversal no and yes
      #find the right traversal no and yes for
      #calculate the entropy of each branch
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

      #calculate the entropy of a branch if there are values to split on
      if (num_00 + num_01 > 0):
        n_entropy = calc_entropy(num_00, num_01, (num_00 + num_01))
      if (num_10 + num_11 > 0):
        y_entropy = calc_entropy(num_10, num_11, (num_10 + num_11))
      #print(num_00, num_01, num_10, num_11, n_entropy, y_entropy, len(cross_index))
      if(len(cross_index) > 1):
        h_node = ((((num_00 + num_01) / len(cross_index)) * n_entropy) +
                  (((num_10 + num_11) / len(cross_index)) * y_entropy))

        #calculate the Information gain and if the IG is better than the current one, use the current node
        #and store the values to create the node
        IG = curr_node.h_left - h_node
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
      """ Computer the Right side now """
      right_node = Node(-1, None, None, x, None, None)
      right_node.copy_path(curr_node.path)
      # append the current path taken to the curr node and add a 1 since we will be going right
      right_node.append_path((curr_node.feature, 1))
      n_00 = 0
      n_01 = 0
      n_10 = 0
      n_11 = 0
      c_index = []
      #find which samples can be used for the current path and feature
      for a in range(labels.shape[0]):
        c_index.append(a)
      c_index_copy = copy.copy(c_index)
      # find which samples can be used for the current path and feature
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
      # for the current samples left, find the left traversal no and yes
      # find the right traversal no and yes for
      # calculate the entropy of each branch
      for c in c_index:
        if (features[c][x] == 0 and labels[c] == 0):
          n_00 = n_00 + 1
        if (features[c][x] == 0 and labels[c] == 1):
          n_01 = n_01 + 1
        if (features[c][x] == 1 and labels[c] == 0):
          n_10 = n_10 + 1
        if (features[c][x] == 1 and labels[c] == 1):
          n_11 = n_11 + 1

      # calculate the entropy of a branch if there are values to split on
      if (n_00 + n_01 > 0):
        rn_entropy = calc_entropy(n_00, n_01, (n_00 + n_01))
      if (n_10 + n_11 > 0):
        ry_entropy = calc_entropy(n_10, n_11, (n_10 + n_11))
      if(len(c_index) > 1):
        h_right = ((((n_00 + n_01) / len(c_index)) * rn_entropy) +
                  (((n_10 + n_11) / len(c_index)) * ry_entropy))

        # calculate the Information gain and if the IG is better than the current one, use the current node
        # and store the values to create the node
        R_IG = curr_node.h_right - h_right
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


  #recursion_left if there is actually something left to split on and check for, meaning that the entropy has been
  #changed to our data type and wthe IG is not 0. Set the left node values that were split and then
  #go ahead and recrusively check the left subtree and build it.
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

  #recursion_right if there is actually something left to split on and check for, meaning that the entropy has been
  #changed to our data type and wthe IG is not 0. Set the right node values that were split and then
  #go ahead and recrusively check the right subtree and build it.
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


""" find_root
    finds the root of the tree by iterating through all of the features and taking the best feature with the 
    best information gain from the labels.
"""
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

    #calculate the entropy of each possible feature for the root node and split.
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
    #If the IG gain is greater than the current one, default set to -inf, then set this as the root node to split on
    #and return the appropriate values
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

""" calc_entropy: 
    calculates the entropy of a given split based on the amount of 0s and 1s passed in and the total labels for that
    split 
"""
def calc_entropy(n, y, total):
  if(n == 0 and y > 0):
    return - 0 - (((y/total) * math.log(y/total, 2)))
  elif(y == 0 and n > 0):
    return -((n / total) * math.log(n/total, 2))- 0
  else:
    return -((n / total) * math.log(n/total, 2)) - ((y/total) * math.log(y/total, 2))
