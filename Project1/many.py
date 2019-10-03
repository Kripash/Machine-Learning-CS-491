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
#print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
tree = dt.DT_train_binary(X, Y, max_depth)
#tree.debug()
accuracy = dt.DT_test_binary(X_test, Y_test, tree)
#print(accuracy)
#print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
acc_tree = dt.DT_train_binary_best(X, Y, X_val, Y_val)
#acc_tree.debug()
acc_accuracy = dt.DT_test_binary(X_test, Y_test, acc_tree)
#print(accuracy)
#print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")



#print("TREE 2")

X_2 = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0]])

Y_2 = np.array([[0], [1], [0], [0], [1], [0], [1], [1], [1]])

tx_2 = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0]])
ty_2 = np.array([[1], [1], [0], [0], [1], [0], [1], [1], [1]])

vx_2 = np.array([[1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 0]])
vy_2 = np.array([[0], [0], [1], [0], [1], [1]])

tree_2 = dt.DT_train_binary(X_2, Y_2, 3)
#tree_2.debug()
accuracy_2 = dt.DT_test_binary(tx_2, ty_2, tree_2)
#print("accuracy is: ", accuracy_2)
#print("***************************************************")
best_train_tree = dt.DT_train_binary_best(X_2, Y_2,vx_2, vy_2)
#best_train_tree.debug()
best_accuracy = dt.DT_test_binary(tx_2, ty_2, best_train_tree)
#print("best accuracy is: ", best_accuracy)

#print("******************************************************************************************")
set_x1 = np.array([[1, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1], [0, 1, 0, 0, 1, 0, 0], [1, 1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 1, 1, 1]])
set_y1 = np.array([[1], [1], [0], [1], [0]])

set_x2 = np.array([[0, 0, 1, 1, 0, 1, 1], [0, 1, 0, 0, 1, 0, 0], [1, 1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 0, 1]])
set_y2 = np.array([[1], [0], [1], [0], [1]])

set_y3 = np.array([[1], [0], [1], [1], [0]])
set_x3 = np.array([[1, 1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 0, 1], [1, 1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 1, 1]])

test_set_x = np.array([[0, 1, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1], [1, 0, 0, 1, 1, 0, 0]])
test_set_y = np.array([[0], [1], [0]])

max = 5
set_tree1 = dt.DT_train_binary(set_x1, set_y1, 5)
#set_tree1.debug()
accuracy_tree1 = dt.DT_test_binary(test_set_x, test_set_y, set_tree1)
#print(accuracy_tree1)
#print("******************************************************************************************")

set_tree2 = dt.DT_train_binary(set_x2, set_y2, 5)
#set_tree2.debug()
accuracy_tree2 = dt.DT_test_binary(test_set_x, test_set_y, set_tree2)
#print(accuracy_tree2)
#print("******************************************************************************************")



set_tree3 = dt.DT_train_binary(set_x3, set_y3, 5)
set_tree3.debug()
accuracy_tree3 = dt.DT_test_binary(test_set_x, test_set_y, set_tree3)
print(accuracy_tree3)
#print("******************************************************************************************")



for votes in range(test_set_y.shape[0]):
  prediction = dt.DT_make_prediction(test_set_x[votes], set_tree3)
  #print("sample : ", votes, " = " , prediction)
