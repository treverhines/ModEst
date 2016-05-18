#!/usr/bin/env python
# Generalize Cross Validation
import numpy as np
from modest.timing import funtime
from numpy.linalg import pinv
from numpy.linalg import inv
import scipy.optimize
import matplotlib.pyplot as plt

def product_trace(A,B):
  ''' 
  efficiently returns the trace of a matrix product
  '''
  return np.sum(A*B.T)


def gcv_predictive_error(alpha,A,L,data):
  ''' 
  solves 

    |   A     | m = | data |
    | alpha*L |   = |  0   |

  for m, where A is the system matrix and L is the regularization 
  matrix.  Then this function returns the predictive error of the 
  solution using generalized cross validation
  '''
  # map alpha to an entirely positive domain
  # alpha = 10**log_alpha

  # compute generalized inverse
  try:
    # compute the inverse with the pseudo inverse, which is more 
    # stable because singular values are removed
    ATA = A.T.dot(A)
    LTL = alpha**2*L.T.dot(L)
    A_ginv = pinv(ATA + LTL).dot(A.T)
  except np.linalg.linalg.LinAlgError:
    # if the generalized inverse cant be computed then the predictive
    # error is infinite
    print('WARNING: failed to compute generalized inverse with damping parameter %s' % alpha)
    return np.inf

  # estimate m
  m = A_ginv.dot(data)

  # compute misfit
  predicted = A.dot(m)
  residual = predicted - data
  misfit = residual.dot(residual)

  # compute predictive error
  numerator = len(data)*misfit
  denominator = (len(data) - product_trace(A,A_ginv))**2
  return numerator/denominator


def cv_predictive_error(alpha,A,L,data):
  ''' 
  returns the predictive error using cross validation
  '''
  N,M = A.shape  
  K = L.shape[0]
  residual = np.zeros(N)
  # allocate an array for stacked A and L matrices
  A_ext = np.zeros((N-1+K,M))
  A_ext[N-1:,:] = alpha*L
  data_ext = np.zeros(N+K-1)
  for removed_idx in range(N):
    kept_idx = np.delete(np.arange(N),removed_idx)
    A_ext[:N-1,:] = A[kept_idx,:]
    data_ext[:N-1] = data[kept_idx]
    # estimated model with remove data point
    mest = np.linalg.lstsq(A_ext,data_ext)[0]
    # prediction for the removed data point
    pred = A[removed_idx,:].dot(mest)
    residual[removed_idx] = pred - data[removed_idx]     

  return residual.dot(residual)/N

@funtime
def predictive_error(alpha,A,L,data,gcv=True):
  if gcv:
    return gcv_predictive_error(alpha,A,L,data)
  else:
    return cv_predictive_error(alpha,A,L,data)


def _log_predictive_error(log_alpha,A,L,data,gcv=True):
  return predictive_error(10**log_alpha,A,L,data,gcv=gcv)


@funtime
def optimal_damping(A,L,data,plot=False,gcv=False,log_bounds=None,itr=100):
  ''' 
  returns the optimal penalty parameter for regularized least squares 
  using generalized cross validation
  
  Parameters
  ----------
    A: (N,M) system matrix
    L: (K,M) regularization matrix
    data: (N,) data vector
    plot: whether to plot the predictive error curve

  '''
  if log_bounds is None:
    log_bounds = (-6.0,6.0)

  alpha_range = 10**np.linspace(log_bounds[0],log_bounds[1],itr)
  # predictive error for all tested damping parameters
  err = np.array([predictive_error(a,A,L,data,gcv=gcv) for a in alpha_range])
  optimal_alpha = alpha_range[np.argmin(err)]
  optimal_err = np.min(err)
  if plot:
    fig,ax = plt.subplots()
    if gcv:
      ax.set_title('GCV curve')
    else:
      ax.set_title('CV curve')
    ax.set_ylabel('predictive error')
    ax.set_xlabel('penalty parameter')
    ax.loglog(alpha_range,err,'k-')
    ax.loglog(optimal_alpha,optimal_err,'ko',markersize=10) 
    ax.grid()
    
  return optimal_alpha
      
