import numpy as np
import clustering as cl


# clustering1.txt
# X = np.array([[0], [1], [2], [7], [8], [9], [12], [14], [15]])
# K = 3
# C = cl.K_Means(X, K)
# print(C)

# clustering2.txt
# X = np.array([[1, 0], [7, 4], [9, 6], [2, 1], [4, 8], [0, 3], [13, 5], [6, 8], [7, 3], [3, 6], [2, 1], [8, 3], [10, 2], [3, 5], [5, 1], [1, 9], [10, 3], [4, 1], [6, 6], [2, 2]])
# # K = 2
# K = 3
# C = cl.K_Means(X, K)
# print(C)


# Better k means

# clustering1.txt
# X = np.array([[0], [1], [2], [7], [8], [9], [12], [14], [15]])
# K = 3
# C = cl.K_Means_better(X, K)
# print(C)

# # clustering2.txt
X = np.array([[1, 0], [7, 4], [9, 6], [2, 1], [4, 8], [0, 3], [13, 5], [6, 8], [7, 3], [3, 6], [2, 1], [8, 3], [10, 2], [3, 5], [5, 1], [1, 9], [10, 3], [4, 1], [6, 6], [2, 2]])
# K = 2
K = 3
C = cl.K_Means_better(X, K)
print(C)
