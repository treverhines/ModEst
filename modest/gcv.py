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


def predictive_error(alpha,A,L,data):
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
    A_ginv = inv(ATA + LTL).dot(A.T)
    #A_ginv = pinv(ATA + LTL,rcond=1e-12).dot(A.T)
  except np.linalg.linalg.LinAlgError:
    # if the generalized inverse cant be computed then the predictive
    # error is infinite
    print('WARNING: failed to compute generalized inverse')
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


def _log_predictive_error(log_alpha,A,L,data):
  return predictive_error(10**log_alpha,A,L,data)


def optimal_damping(A,L,data,rcond=1e-12,plot=False):
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
  # search in logspace
  optimal_logalpha = scipy.optimize.minimize_scalar(
                       _log_predictive_error,
                       args=(A,L,data)).x
  optimal_alpha = 10**optimal_logalpha

  if plot:
    optimal_err = predictive_error(optimal_alpha,A,L,data)

    # test damping parameters in log space
    logalpha = np.log10(optimal_alpha)
    logalpha_min = logalpha - 3  
    logalpha_max = logalpha + 3  
    alpha_range = 10**np.linspace(logalpha_min,logalpha_max,100)
    # predictive error for all tested damping parameters
    err = np.array([predictive_error(a,A,L,data) for a in alpha_range])

    current_ax = plt.gca()
    fig,ax = plt.subplots()
    ax.set_title('GCV curve')
    ax.set_ylabel('predictive error')
    ax.set_xlabel('penalty parameter')
    ax.loglog(alpha_range,err,'k-')
    ax.loglog(optimal_alpha,optimal_err,'ko',markersize=10) 
    ax.grid()
    plt.sca(current_ax)
    
  return optimal_alpha
      
