from sklearn.decomposition import FastICA,PCA
# Hankelization
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
  '''Given a Hankel matrix inverse transforms it to return the input matrix'''
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

######################################################################################################
######################################################################################################
def prebuiltICA(X,num_comps):
    from sklearn.decomposition import FastICA # type: ignore
    ica = FastICA(n_components=num_comps)
    XICA = ica.fit_transform(X)
    return XICA.T

def prebuiltPCA(X,num_comps):
    from sklearn.decomposition import PCA # type: ignore
    pca = PCA(n_components=num_comps)
    XPCA = pca.fit_transform(X)
    return XPCA.T

def innerProduct(x,y):
    from numpy import dot, sum
    return sum(dot(y,x))

def absInnerProduct(data,component):

    return abs(innerProduct(data,component))
######################################################################################################
######################################################################################################

def normalize(inp):
    small = min(inp)
    big = max(inp)
    outp = (inp-small)/(big-small)
    return outp

def innerProducts(X,Y):
        from numpy import dot, sum,array
        outs = []
        for y in Y:
              outs.append(sum(dot(y,X)))
        scores = normalize(outs)
        scores = array([scores,scores])
        return scores

def expDot(X,components):
    import numpy as np
    return np.sum(np.abs(np.dot(components,X)),axis = 1)


def norm2(X):
    import numpy as np
    tot = np.sum(X)
    return X/tot

def nonCluster(X):
    return X