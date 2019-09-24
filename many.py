import decision_trees as dt
import numpy as np

X = np.array([[0,1],[0,0],[1,0],[0,0],[1,1]])
Y = np.array([[1], [0], [0], [0], [1]])


X_val = np.array([[0,0],[0,1],[1,0],[1,1]])
Y_val = np.array([[0],[1],[0],[1]])

X_test = np.array([[0,0],[0,1],[1,0],[1,1]])
Y_test = np.array([[1],[1],[0],[1]])


max_depth = -1

print(X)
print(Y)
tree = dt.DT_train_binary(X, Y, max_depth)
tree.debug()
accuracy = dt.DT_test_binary(X_test, Y_test, tree)
print(accuracy)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
acc_tree = dt.DT_train_binary_best(X, Y, X_val, Y_val)
acc_tree.debug()
acc_accuracy = dt.DT_test_binary(X_test, Y_test, acc_tree)
print(accuracy)