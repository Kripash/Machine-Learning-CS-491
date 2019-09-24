import decision_trees as dt
import numpy as np

X = np.array([[0,1],[0,0],[1,0],[0,0],[1,1]])
Y = np.array([[1], [0], [0], [0], [1]])


X_val = np.array([[0,0],[0,1],[1,0],[1,1]])
Y_val = np.array([[0],[1],[0],[1]])

X_test = np.array([[0,0],[0,1],[1,0],[1,1]])
Y_test = np.array([[1],[1],[0],[1]])


max_depth = -1


#print(X)
#print(Y)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
tree = dt.DT_train_binary(X, Y, max_depth)
tree.debug()
accuracy = dt.DT_test_binary(X_test, Y_test, tree)
print(accuracy)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
acc_tree = dt.DT_train_binary_best(X, Y, X_val, Y_val)
acc_tree.debug()
acc_accuracy = dt.DT_test_binary(X_test, Y_test, acc_tree)
print(accuracy)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")



print("TREE 2")

X_2 = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0]])

Y_2 = np.array([[0], [1], [0], [0], [1], [0], [1], [1], [1]])

tx_2 = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0]])
ty_2 = np.array([[1], [1], [0], [0], [1], [0], [1], [1], [1]])

tree_2 = dt.DT_train_binary(X_2, Y_2, 3)
tree_2.debug()
accuracy_2 = dt.DT_test_binary(tx_2, ty_2, tree_2)
print("accuracy is: ", accuracy_2)

train_accuracy = dt.DT_test_binary(X_2, Y_2, tree_2)
print("training on tree: ", train_accuracy)