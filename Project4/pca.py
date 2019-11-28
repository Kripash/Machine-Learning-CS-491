import numpy as np
import matplotlib.pyplot as plt
import copy

debug = False

def compute_Z(X, centering=True, scaling=False):
  global debug
  Z = copy.copy(X)
  mean = np.mean(Z, axis=0)
  std = np.std(Z, axis=0)
  if debug:
    print("Mean: ", mean)
    print("Std: ", std)
  if(centering):
    Z = Z - mean
  if debug:
    print("Z: \n", Z)
  if(scaling):
    Z = Z/std
  if debug:
    print("Scaling: \n", Z)

  return Z


def compute_covariance_matrix(Z):
  global debug
  Z_T = np.transpose(Z)
  covariance_z = np.dot(Z_T, Z)
  if debug:
    print("covariance_z: \n", covariance_z)
  return covariance_z


def find_pcs(COV):
  global debug
  L, PCS = np.linalg.eig(COV)
  L = L.real 
  PCS = PCS.real
  if debug:
    print("eigen values: \n", L)
    print("eigen vectors: \n", PCS)

  indices = L[::-1].argsort()
  L[::-1].sort()
  if debug:
    print("Sorted Eigen Values: \n", L)
    print("Sorted Eigen Vectors: \n", PCS)
    print("Indices: \n", indices)

  PCS = PCS.T[indices].T
  if debug:
    print("PCS: \n", PCS)

  return L, PCS


def determineK(L, var):
  sum = np.sum(L)
  for i in range(len(L)):
    variance = np.sum(L[:i]) / sum
    if(variance >= var):
      return i 
  return len(L)

def project_data(Z, PCS, L, k, var):
  if(k == 0):
    k = determineK(L, var)

  if debug:
    print("L: ", k)
  U = PCS[:, :k]
  if debug:
    print("U: \n", U)
    print("Z: \n", Z)
  projection = np.dot(Z, U)

  if debug:
    print("projection: \n", projection)

  return projection



"""X = np.array([[1, -1], [-1, 1], [1, -1], [1,1]])
Z = compute_Z(X, centering=True, scaling=True)
cov_z = compute_covariance_matrix(Z)
L, PCS = find_pcs(cov_z)
print(PCS)
Z_star = project_data(Z, PCS, L, 1, 0)"""
