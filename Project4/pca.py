import numpy as np
import matplotlib.pyplot as plt
import copy


def compute_Z(X, centering=True, scaling=False):
  Z = copy.copy(X)
  mean = np.mean(Z, axis=0)
  std = np.std(Z, axis=0)

  if(centering):
    Z = Z - mean
 
  if(scaling):
    Z = Z/std

  return Z


def compute_covariance_matrix(Z):
  Z_T = np.transpose(Z)
  covariance_z = np.dot(Z_T, Z)

  return covariance_z


def find_pcs(COV):
  L, PCS = np.linalg.eig(COV)
  L = L.real 
  PCS = PCS.real

  indices = np.flip( L.argsort() ) # indices now has indices of eigenvalues in the order of largest to smallest.
  #print(indices)
  L = np.flip( L.sort() )

  PCS = PCS[:, indices]

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

  U = PCS[:, :k]

  projection = np.dot(Z, U)

  return projection


# X = np.array([[1, -1], [-1, 1], [1, -1], [1,1]])
# Z = compute_Z(X, centering=True, scaling=True)
# cov_z = compute_covariance_matrix(Z)
# L, PCS = find_pcs(cov_z)
# print(PCS)
# Z_star = project_data(Z, PCS, L, 1, 0)
