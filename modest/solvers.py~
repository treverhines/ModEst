#!/usr/bin/env python
'''
routines which solve min_m||Gm - d|| with potentially additional
constraints on m.
'''
import numpy as np
import scipy.optimize
import scipy.sparse.linalg
import scipy.linalg
import scipy.sparse
import modest._bvls as _bvls
import logging
try:
  import modest.petsc
except ImportError:
  print('cannot import PETSc. some solvers will not work properly')

logger = logging.getLogger(__name__)

def _arg_checker(fin):
  def fout(G,d,*args,**kwargs):
    G = np.asarray(G)
    d = np.asarray(d)
    if len(d.shape) > 1:
      d = np.squeeze(d)    

    assert len(d.shape) == 1, 'data vector must be 1 dimensional'
    assert len(G.shape) == 2, 'system matrix must be 2 dimensional'
    assert G.shape[0] == d.shape[0], 'data vector and system matrix have mismatched dimensions'
    assert np.isfinite(G).all(), 'system matrix has non finite values'
    assert np.isfinite(d).all(), 'data vector has non finite values'
    output = fin(G,d,*args,**kwargs)
    assert len(output) == G.shape[1], 'solver returned a solution with incorrect size'
    return output

  fout.__doc__ = fin.__doc__
  fout.__name__ = fin.__name__
  return fout

def _reg_arg_checker(fin):
  def fout(G,L,d,*args,**kwargs):
    G_is_sparse = scipy.sparse.isspmatrix(G)
    L_is_sparse = scipy.sparse.isspmatrix(L)
    if not G_is_sparse:
      G = np.asarray(G)

    if not L_is_sparse:
      L = np.asarray(L)

    d = np.asarray(d)
    if len(d.shape) > 1:
      d = np.squeeze(d)    

    assert len(d.shape) == 1, 'data vector must be 1 dimensional'
    assert len(G.shape) == 2, 'system matrix must be 2 dimensional'
    assert len(L.shape) == 2, 'regularization matrix must be 2 dimensional'
    assert G.shape[0] == d.shape[0], 'data vector and system matrix have mismatched dimensions'
    assert G.shape[1] == L.shape[1], 'columns in the system matrix must equal to columns in the regularization matrix'
    output = fin(G,L,d,*args,**kwargs)
    assert len(output) == G.shape[1], 'solver returned a solution with incorrect size'
    return output

  fout.__doc__ = fin.__doc__
  fout.__name__ = fin.__name__
  return fout


# Functions that solve Gm = d in a least squares sense
#####################################################################
@_arg_checker
def lstsq(G,d,*args,**kwargs):
  '''                                     
  wrapper from scipy.linalg.lstsq  
  '''
  out = scipy.linalg.lstsq(G,d,*args,**kwargs)[0]
  #out = np.linalg.lstsq(G,d,*args,**kwargs)[0]
  return out

@_arg_checker
def nnls(G,d,*args,**kwargs):
  '''               
  wrapper from scipy.optimize.nnls
  '''
  out = scipy.optimize.nnls(G,d,*args,**kwargs)[0]
  return out

@_arg_checker
def bvls(G,d,lower_lim,upper_lim):
  '''                                                                                         
  Bounded Value Least Squares
                                                               
  This is a python wrapper of the Fortran90 bvls module originally    
  written by Charles Lawson and Richard Hanson and then modified by John   
  Burkardt. 
                                                                                 
  PARAMETERS
  ----------
    G : (M,N) system matrix mapping the model vector to observation vector
    d : (M,) observation vector
    lower_bounds : (N,) vector for the lower bounds on the model vector
    upper_bounds : (N,) vector for the upper bounds on the model vector

  RETURNS
  -------
    m : (N,) vector that satisfies min(||Gm - d||) subject to the imposed
        constraints

  USAGE             
  -----              
    >>>import bvls                             
    >>>G = np.random.random((10,2))      
    >>>m = np.array([1.0,2.0])         
    >>>d = G.dot(m)                        
    >>>lower_bounds = np.array([0.0,0.0])           
    >>>upper_bounds = np.array([1.5,1.5])           
    >>>output = bvls(G,d,lower_bounds,upper_bounds) 
    >>>m_est = output[0]               
  '''
  bounds = np.array([lower_lim,upper_lim])
  (soln,rnorm,nstep,w,index,err) = _bvls.bvls(G,d,bounds)
  if err == 0:
    logger.debug('exit status 0: Solution completed')

  if err == 1:
    print('exit status 1: M  <=  0 or N  <=  0 ')
    logger.debug('exit status 1: M  <=  0 or N  <=  0 ')

  if err == 2:
    print('exit status 2: B(:), X(:), BND(:,:), W(:), or INDEX(:) size or shape violation')
    logger.debug('exit status 2: B(:), X(:), BND(:,:), W(:), or INDEX(:) size or shape violation')

  if err == 3:
    print('exit status 3: Input bounds are inconsistent')
    logger.warning('exit status 3: Input bounds are inconsistent')

  if err == 4:
    print('exit status 4: Exceed maximum number of iterations')
    logger.warning('exit status 4: Exceed maximum number of iterations')

  return soln

@_arg_checker
def dsolve(G,d,*args,**kwargs):
  ''' 
  multiply the lhs and rhs by G.T and then use a direct solver
  '''
  return scipy.linalg.solve(G.T.dot(G),G.T.dot(d),sym_pos=True,*args,**kwargs)

# Functions that solve Gm = d in a least squares sense with 
# regularization constraints
#####################################################################
@_reg_arg_checker
def reg_dsolve(A,L,data):
  lhs = A.T.dot(A) + L.T.dot(L)
  rhs = A.T.dot(data)
  return np.linalg.solve(lhs,rhs)

@_reg_arg_checker
def sparse_reg_dsolve(A,L,data,**kwargs):
  ''' 
  solves the least squares problem with LU factorization
  '''
  A = A.tocsr()
  L = L.tocsr()
  lhs = A.T.dot(A) + L.T.dot(L)
  rhs = A.T.dot(data)
  soln = scipy.sparse.linalg.spsolve(lhs,rhs,**kwargs)
  return soln

@_reg_arg_checker
def sparse_reg_petsc(A,L,data,**kwargs):
  ''' 
  solves the least squares problem with petsc. The left-hand side is 
  first converted to a square matrix. arguments are passed to 
  modest.petsc.petsc_solve
  '''
  A = A.tocsr()
  L = L.tocsr()
  lhs = A.T.dot(A) + L.T.dot(L)
  rhs = A.T.dot(data)

  soln = modest.petsc.petsc_solve(lhs,rhs,**kwargs)
  return soln

@_reg_arg_checker
def sparse_reg_lgmres(A,L,data,**kwargs):
  ''' 
  solves the least squares problem with lmgres. The left-hand side is 
  first converted to a square matrix
  '''
  A = A.tocsr()
  L = L.tocsr()
  lhs = A.T.dot(A) + L.T.dot(L)
  rhs = A.T.dot(data)

  soln,info = scipy.sparse.linalg.lgmres(lhs,rhs,**kwargs)
  if info < 0:
    logger.warning('LGMRES exited with value %s' % info)

  elif info == 0:
    logger.info('LGMRES finished successfully')

  elif info > 0:
    print('WARNING: LGMRES did not converge after %s iterations' % info)
    logger.warning('LGMRES did not converge after %s iterations' % info)

  return soln

@_reg_arg_checker
def sparse_reg_lsqr(A,L,data,**kwargs):
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
  out = scipy.sparse.linalg.lsqr(A_ext,data_ext,**kwargs)
  soln = out[0]
  istop = out[1]
  if istop not in [1,2]:
    print('WARNING: %s' % msg[istop])
    logger.warning(msg[istop])

  else:
    logger.info(msg[istop])

  return soln

