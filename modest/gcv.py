#!/usr/bin/env python
# Generalize Cross Validation
import numpy as np 
from numpy.linalg import pinv 
from numpy.linalg import inv 
from scipy.sparse import isspmatrix 
import scipy.sparse.linalg as spla
import scipy.optimize 
import matplotlib.pyplot as plt 
import logging 
import rbf.halton

logger = logging.getLogger(__name__)


def dense_direct_solve(A,L,data):
  lhs = A.T.dot(A) + L.T.dot(L)
  rhs = A.T.dot(data)
  return np.linalg.solve(lhs,rhs)


def sparse_direct_solve(A,L,data,**kwargs):
  ''' 
  solves the least squares problem with LU factorization
  '''
  A = A.tocsr()
  L = L.tocsr()
  lhs = A.T.dot(A) + L.T.dot(L)
  rhs = A.T.dot(data) 
  soln = spla.spsolve(lhs,rhs,**kwargs)
  return soln


def sparse_iter_solve(A,L,data,**kwargs):
  ''' 
  solves the least squares problem with lsqr
  '''
  msg = {0:'The exact solution is  x = 0',
         1:'Ax - b is small enough, given atol, btol',
         2:'The least-squares solution is good enough, given atol',
         3:'The estimate of cond(Abar) has exceeded conlim',
         4:'Ax - b is small enough for this machine',
         5:'The least-squares solution is good enough for this machine',
         6:'Cond(Abar) seems to be too large for this machine',
         7:'The iteration limit has been reached',
         8:'The truncated direct error is small enough, given etol'}

  A = A.tocsr()
  L = L.tocsr()
  A_ext = scipy.sparse.vstack((A,L))
  data_ext = np.concatenate((data,np.zeros(L.shape[0])))
  out = spla.lsqr(A_ext,data_ext,**kwargs)
  soln = out[0]
  istop = out[1]
  if istop not in [1,2]:
    print('WARNING: %s' % msg[istop])
    logger.warning(msg[istop])

  else:
    logger.info(msg[istop])

  return soln


def chunkify(list,N):
    return [list[i::N] for i in range(N)]


def product_trace(A,B):
  ''' 
  efficiently returns the trace of a matrix product
  '''
  return np.sum(A*B.T)


def gcv_predictive_error(alpha,A,L,data):
  ''' 
  DEPRICATED 
 
  solves 

    |   A     | m = | data |
    | alpha*L |   = |  0   |

  for m, where A is the system matrix and L is the regularization 
  matrix.  Then this function returns the predictive error of the 
  solution using generalized cross validation
  '''
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


def dense_cv_predictive_error(damping,A,L,data,fold=10):
  ''' 
  Description
  -----------
    Computes the predictive error for the given damping parameter(s).
  
  Parameters
  ----------
    damping: damping parameter or list of damping parameters for each 
      regularization matrix in L

    A: (N,M) dense system matrix

    L: (K,M) regularization matrix or list of regularization matrices

    data: (N,) data array

    fold (default=10): number of cross validation folds
   
    dsolve (default=True): use a direct LU factorization to solve. 
      An iterative solve is used otherwise.

  Note
  ----
    additional key word arguments are passed to the solver
  '''
  # make input a list if it is not already
  if not hasattr(damping,'__iter__'):
    damping = [damping]
    L = [L]

  if len(damping) != len(L):
    raise ValueError(
      'number of damping parameters must equal number of '
      'regularization matrices')

  if not all(np.isscalar(d) for d in damping):
    raise TypeError(
      'damping must be a scalar or list of scalars')

  # scale regularization matrices. note that this makes copies 
  L = [k*d for d,k in zip(damping,L)]

  # stack regularization matrices
  L = np.vstack(L)

  N = A.shape[0]  
  K = L.shape[0]
  # make sure folds is smaller than the number of data points
  fold = min(fold,N)

  res = np.zeros(N)
  for rmidx in chunkify(range(N),fold):
    # build weight matrix. data points are excluded by given them zero 
    # weight
    weight = np.ones(N)
    weight[rmidx] = 0.0
    
    A_new = weight[:,None]*A
    data_new = weight*data

    soln = dense_direct_solve(A_new,L,data_new)
    pred = A.dot(soln)

    res[rmidx] = pred[rmidx] - data[rmidx]     

  return res.dot(res)/N


def sparse_cv_predictive_error(damping,A,L,data,fold=10,dsolve=True,**kwargs):
  ''' 
  Description
  -----------
    Computes the predictive error for the given damping parameter(s).
  
  Parameters
  ----------
    damping: damping parameter or list of damping parameters for each 
      regularization matrix in L

    A: (N,M) sparse system matrix

    L: (K,M) regularization matrix or list of regularization matrices

    data: (N,) data array

    fold (default=10): number of cross validation folds
   
    dsolve (default=True): use a direct LU factorization to solve. 
      An iterative solve is used otherwise.

  Note
  ----
    additional key word arguments are passed to the solver
  '''
  # make input a list if it is not already
  if not hasattr(damping,'__iter__'):
    damping = [damping]
    L = [L]

  if len(damping) != len(L):
    raise ValueError(
      'number of damping parameters must equal number of '
      'regularization matrices')

  if not all(np.isscalar(d) for d in damping):
    raise TypeError(
      'damping must be a scalar or list of scalars')

  if not all(isspmatrix(k) for k in L):
    raise TypeError(
      'L must be a sparse matrix or list of sparse matrices')

  if not isspmatrix(A):
    raise TypeError(
      'A must be a sparse matrix or list of sparse matrices')
  
  # scale regularization matrices. note that this makes copies 
  L = [k*d for d,k in zip(damping,L)]

  # stack regularization matrices
  L = scipy.sparse.vstack(L)
  
  N = A.shape[0]  
  K = L.shape[0]
  # make sure folds is smaller than the number of data points
  fold = min(fold,N)

  # empty residual vector
  res = np.zeros(N)

  for rmidx in chunkify(range(N),fold):
    # build weight matrix. data points are excluded by given them zero 
    # weight
    diag = np.ones(N)
    diag[rmidx] = 0.0
    W = scipy.sparse.diags(diag,0)
    
    # note that there are multiple matrix copies made here
    A_new = W.dot(A)
    data_new = W.dot(data)

    if dsolve:
      soln = sparse_direct_solve(A_new,L,data_new,**kwargs)
      pred = A.dot(soln)
      
    else:
      soln = sparse_iter_solve(A_new,L,data_new,**kwargs)
      pred = A.dot(soln)

    res[rmidx] = pred[rmidx] - data[rmidx]     

  return res.dot(res)/N


def predictive_error(damping,A,L,data,fold=10,**kwargs):
  if isspmatrix(A):
    return sparse_cv_predictive_error(damping,A,L,data,fold=fold,**kwargs)
  else:
    return dense_cv_predictive_error(damping,A,L,data,fold=fold,**kwargs)


def optimal_damping_parameters(A,L,data,
                               fold=10,log_bounds=None,
                               itr=100,**kwargs):
  ''' 
  returns the optimal penalty parameter for regularized least squares 
  using generalized cross validation
  
  Parameters
  ----------
    A: (N,M) system matrix
    L: list of (K,M) regularization matrices
    data: (N,) data vector
    plot: whether to plot the predictive error curve
    log_bounds: list of lower and upper bounds for each penalty parameter

  '''
  # number of damping parameters
  P = len(L)

  # values range from 0 to 1
  tests = rbf.halton.halton(itr,P)  
  
  # scale tests to the desired bounds
  if log_bounds is None:
    log_bounds = [[-6.0,6.0]]*P

  log_bounds = np.asarray(log_bounds)
  if log_bounds.shape != (P,2):
    raise TypeError('log_bounds must be a length P list of lower and upper bounds')

  bounds_diff = log_bounds[:,1] - log_bounds[:,0]
  bounds_min = log_bounds[:,0]

  tests = tests*bounds_diff
  tests = tests + bounds_min
  tests = 10**tests

  errs = []
  for t in tests:
    errs += [predictive_error(t,A,L,data,fold=fold,**kwargs)]

    best_err = np.min(errs)
    best_idx = np.argmin(errs)
    best_test = tests[best_idx]
    logger.debug('predictive error for damping parameters %s: %s' % (t,errs[0]))
    logger.debug('best predictive error: %s' % best_err)
    logger.debug('best damping parameters: %s' % best_test)

  logger.info('best predictive error: %s' % best_err)
  logger.info('best damping parameters: %s' % best_test)

  return best_test, best_err, tests, np.array(errs)

def optimal_damping_parameter(A,L,data,plot=False,
                              fold=10,log_bounds=None,
                              itr=100,**kwargs):
  ''' 
  used when only searching for one penalty parameter
  '''
  if log_bounds is None:
    log_bounds = [-6.0,6.0]

  out = optimal_damping_parameters(A,[L],data,fold=fold,
                                   log_bounds=[log_bounds],
                                   itr=itr,**kwargs)
  best_test,best_err,tests,errs = out
  best_test = best_test[0]
  tests = tests[:,0]

  if plot:
    # sort for plotting purposes
    sort_idx = np.argsort(tests)
    tests = tests[sort_idx]
    errs = errs[sort_idx]

    fig,ax = plt.subplots()
    ax.set_title('%s-fold cross validation curve' % fold)
    ax.set_ylabel('predictive error')
    ax.set_xlabel('penalty parameter')
    ax.loglog(tests,errs,'k-')
    ax.loglog(best_test,best_err,'ko',markersize=10) 
    ax.grid()
    plt.show()
    
  return best_test, best_err, tests, np.array(errs)
      
