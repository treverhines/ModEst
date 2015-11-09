#!/usr/bin/env python
'''
routines which solve min_m||Gm - d|| with potentially additional
constraints on m.
'''
import numpy as np
import scipy.optimize
import scipy.sparse.linalg
import scipy.linalg
import modest._bvls as _bvls
from modest.converger import Converger
import logging
from modest.timing import funtime
logger = logging.getLogger(__name__)

def _arg_checker(fin):
  def fout(G,d,*args,**kwargs):
    G = np.asarray(G)
    d = np.asarray(d)
    G_shape = np.shape(G)  
    d_shape = np.shape(d)
    if len(d_shape) > 1:
      d = np.squeeze(d)    
      d_shape = np.shape(d)

    assert len(d_shape) == 1
    assert len(G_shape) == 2
    assert G_shape[0] == d_shape[0]
    output = fin(G,d,*args,**kwargs)
    assert len(output) == G_shape[1]
    return output

  fout.__doc__ = fin.__doc__
  fout.__name__ = fin.__name__
  return fout

@funtime
@_arg_checker
def lstsq(G,d,*args,**kwargs):
  '''                                     
  wrapper from scipy.linalg.lstsq  
  '''
  out = scipy.linalg.lstsq(G,d,*args,**kwargs)[0]
  #out = np.linalg.lstsq(G,d,*args,**kwargs)[0]
  return out

@funtime
@_arg_checker
def nnls(G,d,*args,**kwargs):
  '''               
  wrapper from scipy.optimize.nnls
  '''
  out = scipy.optimize.nnls(G,d,*args,**kwargs)[0]
  return out

@funtime
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

@funtime
@_arg_checker
def cgls(G,d,m_o=None,maxitr=2000,rtol=1e-16,atol=1e-16):
  '''
  congugate gradient least squares

  algorithm from Aster et al. 2005
  '''
  N,M = np.shape(G)
  if m_o is None:
    m_o = np.zeros(M)

  s = d - G.dot(m_o)
  p = np.zeros(M)
  r = s.dot(G)
  r_prev = np.zeros(M)
  beta = 0

  conv = Converger(np.zeros(N),atol=atol,rtol=rtol,maxitr=maxitr)
  status,message = conv(s)
  logger.debug(message)
  k = 0
  while (status !=  0) & (status != 3):
    if k > 0:
      beta = r.dot(r)/r_prev.dot(r_prev)
    p = r + beta*p
    Gp = G.dot(p)
    alpha = r.dot(r)/Gp.dot(Gp)
    #print('hi')
    #alpha = r.dot(r)/Gp.dot(Gp)
    m_o = m_o + alpha*p
    s = s - alpha*Gp
    r_prev[:] = r
    r = s.dot(G)
    status,message = conv(s)
    conv.set(s)
    logger.debug(message)
    k += 1

  return m_o

@funtime
@_arg_checker
def cg(G,d,*args,**kwargs):
  '''
  solves GtG = Gtd using scipy's cg solver. This tends to be
  about as fast as cgls
  '''
  GtG = G.transpose().dot(G)
  Gtd = G.transpose().dot(d)
  return scipy.sparse.linalg.cg(GtG,Gtd,*args,**kwargs)[0]



