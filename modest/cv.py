#!/usr/bin/env python
# Generalize Cross Validation
import numpy as np 
from scipy.sparse import isspmatrix 
import scipy.optimize 
import matplotlib.pyplot as plt 
import matplotlib.tri as tri
import logging 
import rbf.halton
from myplot.cm import viridis
import modest.petsc
import modest.solvers
from myplot.colorbar import pseudo_transparent_cmap

logger = logging.getLogger(__name__)


def chunkify(list,N):
    # make chunks random
    list = np.asarray(list)
    K = list.shape[0]

    # I need a randomly mixed list, but i need to to be mixed the 
    # same time every time
    current_random_state = np.random.get_state()
    np.random.seed(1)
    mix_range = np.random.choice(range(K),K,replace=False)
    np.random.set_state(current_random_state)    

    chunked_mix_range = [mix_range[i::N] for i in range(N)]
    return [list[c] for c in chunked_mix_range]


def dense_predictive_error(damping,A,L,data,fold=10):
  ''' 
  Description
  -----------
    Computes the predictive error for the given damping parameter(s).
  
  Parameters
  ----------
    damping: list of damping parameters for each regularization matrix 

    A: (N,M) dense system matrix

    L: list of regularization matrices

    data: (N,) data array

    fold (default=10): number of cross validation folds

  '''
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

    soln = modest.solvers.reg_ds(A_new,L,data_new)
    pred = A.dot(soln)

    res[rmidx] = pred[rmidx] - data[rmidx]     

  return res.dot(res)/N


def sparse_predictive_error(damping,A,L,data,fold=10,solver='spsolve',**kwargs):
  ''' 
  Description
  -----------
    Computes the predictive error for the given damping parameter(s).
  
  Parameters
  ----------
    damping: list of damping parameters for each regularization matrix 

    A: (N,M) sparse system matrix

    L: list of regularization matrices

    data: (N,) data array

    fold (default=10): number of cross validation folds
   
    solver: which solver to use. choices are
      'spsolve': scipy.spares.linalg.spsolve
      'lgmres': scipy.sparse.linalg.lgmres
      'lsqr': scipy.sparse.linalg.lsqr
      'petsc': modest.petsc.petsc_solve  

      additional key word arguments are passed to the solver
    
  '''
  solver_dict = {'spsolve':modest.solvers.sparse_reg_ds,
                 'lgmres':modest.solvers.sparse_reg_lgmres,
                 'lsqr':modest.solvers.sparse_reg_lsqr,
                 'petsc':modest.solvers.sparse_reg_petsc}
 
  if len(damping) != len(L):
    raise ValueError(
      'number of damping parameters must equal number of '
      'regularization matrices')

  if not all(np.isscalar(d) for d in damping):
    raise TypeError(
      'damping must be a list of scalars')

  if not all(isspmatrix(k) for k in L):
    raise TypeError(
      'L must be a list of sparse matrices')

  if not isspmatrix(A):
    raise TypeError(
      'A must be a list of sparse matrices')
  
  # scale regularization matrices. note that this makes copies 
  L = (k*d for d,k in zip(damping,L))

  # stack regularization matrices
  L = scipy.sparse.vstack(L)
  
  N = A.shape[0]  
  # make sure folds is smaller than the number of data points
  fold = min(fold,N)

  # empty residual vector
  res = np.zeros(N)

  for rmidx in chunkify(range(N),fold):
    # build weight matrix. data points are excluded by giving them zero 
    # weight
    diag = np.ones(N)
    diag[rmidx] = 0.0
    W = scipy.sparse.diags(diag,0)
    
    # note that there are multiple matrix copies made here
    A_new = W.dot(A)
    data_new = W.dot(data)

    soln = solver_dict[solver](A_new,L,data_new,**kwargs)
    pred = A.dot(soln)

    res[rmidx] = pred[rmidx] - data[rmidx]     

  return res.dot(res)/N


def predictive_error(damping,A,L,data,fold=10,**kwargs):
  if isspmatrix(A):
    return sparse_predictive_error(damping,A,L,data,fold=fold,**kwargs)
  else:
    return dense_predictive_error(damping,A,L,data,fold=fold,**kwargs)


def optimal_damping_parameters(A,L,data,
                               fold=10,log_bounds=None,
                               itr=100,plot=False,**kwargs):
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

  errs = np.zeros(itr)
  for i,t in enumerate(tests):
    errs[i] = predictive_error(t,A,L,data,fold=fold,**kwargs)
    logger.info('calculated %s of %s predictive errors' % ((i+1),itr))

  best_err = np.min(errs)
  best_idx = np.argmin(errs)
  best_test = tests[best_idx]

  logger.info('best predictive error: %s' % best_err)
  logger.info('best damping parameters: %s' % best_test)

  if (P > 2) & plot:
    logger.info(
      'cannot plot predictive error for more than two damping '
      'parameters')

  elif (P == 1) & plot:
    # sort for plotting purposes
    sort_idx = np.argsort(tests[:,0])
    tests = tests[sort_idx,:]
    errs = errs[sort_idx]

    fig,ax = plt.subplots()
    ax.set_title('%s-fold cross validation curve' % fold)
    ax.set_ylabel('predictive error')
    ax.set_xlabel('penalty parameter')
    ax.loglog(tests[:,0],errs,'k-')
    ax.loglog(best_test[0],best_err,'ko',markersize=10) 
    ax.grid(zorder=-1)
    fig.tight_layout()

  elif (P == 2) & plot:
    fig,ax = plt.subplots()

    log_tests = np.log10(tests)
    log_errs = np.log10(errs)
    vmin = np.min(log_errs)
    vmax =np.max(log_errs)
    viridis_alpha = pseudo_transparent_cmap(viridis,0.5)
    # make triangularization in logspace
    triangles = tri.Triangulation(log_tests[:,0],log_tests[:,1])
    # set triangles to linear space
    triangles.x = tests[:,0]
    triangles.y = tests[:,1]
    ax.tripcolor(triangles,log_errs,
                 vmin=vmin,vmax=vmax,cmap=viridis_alpha,zorder=0)
    ax.scatter([best_test[0]],[best_test[1]],
               s=200,c=[np.log10(best_err)],
               vmin=vmin,vmax=vmax,zorder=2,cmap=viridis)
    c = ax.scatter(tests[:,0],tests[:,1],s=50,c=log_errs,
                   vmin=vmin,vmax=vmax,zorder=1,cmap=viridis)
    ax.set_title('%s-fold cross validation curve' % fold)
    ax.set_xscale('log')
    ax.set_yscale('log')
    cbar = fig.colorbar(c)
    cbar.set_label('log predictive error')
    ax.set_xlabel('damping parameter 1')
    ax.set_ylabel('damping parameter 2')
    ax.grid(zorder=0)
    fig.tight_layout()

  return best_test, best_err, tests, errs


def optimal_damping_parameter(A,L,data,
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
    
  return best_test, best_err, tests, errs

      
