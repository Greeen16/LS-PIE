# FUNCTION 1 - Hankelization

  #construct here the function
def hankelize(F,L):
  '''constructs the L-Trajectory Matrix X given an input (F) and shift length (L)'''
  import numpy as np
  L = int(L)
  N = len(F)
  K = N - L +1
  X  = np.zeros([K,L])
  for i in range(K):
    j = i+1

    X[i] = F[j-1:j+L-2+1]

  return X.T

def inverseHankelize(X):
  import numpy as np
  [L,K] = np.shape(X)
  N = K+L -1
  F = np.zeros(N)
  X = np.array(X)
  X = X.T
  for i in range(K):
    j = i+1
    F[j-1:j+L-2+1] = X[i]

  return F
  
  # FUNCTIONS 2
def prebuiltICA(X,num_comps):
    from sklearn.decomposition import FastICA # type: ignore
    ica = FastICA(n_components=num_comps)
    XICA = ica.fit_transform(X)
    return XICA

def innerProduct(x,y):
    from numpy import dot, sum
    return sum(dot(y,x))