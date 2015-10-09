#!/usr/bin/env python
import numpy as np
import scipy

class Weight:
  def __init__(self,cov=None,var=None,std=None,weight=None): 
    if cov is not None:                
      cov = np.asarray(cov)
      assert axes_no(cov) == 2, 'covariance matrix must be 2 dimensional'
      if isdiagonal(cov):
        self.dim = 1
        self.W = 1.0/np.sqrt(np.diag(cov))                  

      else:
        self.dim = 2 
        N = len(cov)
        A = np.linalg.cholesky(cov)
        W = scipy.linalg.solve_triangular(A,np.eye(N),lower=True)
        self.W = W

    elif var is not None:
      var = np.asarray(var)
      assert axes_no(var) == 1, 'variance must be 1 dimensional'
      self.dim = 1
      self.W = 1.0/np.sqrt(var)

    elif std is not None:
      std = np.asarray(std)
      assert axes_no(std) == 1, 'standard deviation must be 1 dimensional'
      self.dim = 1
      self.W = 1.0/std
    
 
    elif weight is not None:
      weight = np.asarray(weight)
      assert axes_no(weight) == 2, 'weight matrix must be 2 dimensional'
      if isdiagonal(weight):
        self.dim = 1
        self.W = np.diag(weight)

      else:
        self.dim = 2
        self.W = weight
  
  def __call__(self,A):
    A = np.asarray(A)
    N = axes_no(A)
    if self.dim == 1:
      Wext = add_axes(self.W,N-1)
      return Wext*A

    elif self.dim == 2:
      if N <= 2:
        return self.W.dot(A)
      else:
        return np.einsum('ij,j...->i...',self.W,A)    

  def get_array(self):
    if self.dim == 1:
      return np.diag(self.W)

    elif self.dim == 2:
      return self.W


def add_axes(x,N):
  out = np.array(x,copy=True)
  for i in range(N):
    out = out[:,None]

  return out

def axes_no(A):
  return len(np.shape(A))

def isdiagonal(A):
  return np.all(np.diag(np.diag(A)) == A)

def covariance_to_weight(C):
  if is1d(C):
    W = 1.0/np.sqrt(C)

  elif isdiagonal(C):
    W = 1.0/np.sqrt(np.diag(C))

  else:
    N = np.shape(C)[0]
    A = np.linalg.cholesky(C)
    W = scipy.linalg.solve_triangular(A,np.eye(N),lower=True)

  return W
  
