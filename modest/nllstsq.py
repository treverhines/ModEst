#!/usr/bin/env python
import numpy as np
import logging
import modest.solvers as solvers
import modest.timing as timing
from modest.tikhonov import Perturb
from modest.tikhonov import tikhonov_matrix
from modest.timing import funtime
import scipy.sparse

logger = logging.getLogger(__name__)

##------------------------------------------------------------------------------
def make_jacobian(system,dm=1e-4):
  '''
  Description
  -----------
    Creates a finite diffrence Jacobian function

  Parameters
  ----------
    system: callable which takes a vector of model parameters as its 
      first argument 

    dm: step size used to estimate the derivatives

  Returns 
  ------- 
    jacobian: callable which takes a vector of model parameters as its
      first argument and returns an approximation to the Jacobian
      matrix at that point in model space. Additional args and kwargs
      get passed to the system function.

  '''
  def jacobian(m_ref,*args,**kwargs):
    data_ref = system(m_ref,*args,**kwargs)
    data_ref = np.asarray(data_ref)
    param_no = len(m_ref)
    data_no = len(data_ref)
    jac = np.zeros((data_no,param_no))
    for i,m_pert in enumerate(Perturb(m_ref,dm)):
      data_pert = system(m_pert,*args,**kwargs)
      data_pert = np.asarray(data_pert)
      jac[:,i]  = (data_pert - data_ref)/dm

    return jac

  return jacobian

##------------------------------------------------------------------------------
def is1d(A):
  return (len(np.shape(A)) == 1)

def covariance_to_weight(C):
  '''
  Description
  -----------
    Converts a covariance matrix into a weight matrix by using a
    Cholesky decomposition.  If C is a 1D array then the weights will be
    the square root of the reciprocal of C. The output of this function
    will always have the same shape as C
  
    The weight matrix, W, satisfies

      np.linalg.inv(C) = W.T.dot(W)                     (1)

    W is found by using Cholesky factorization of C to find A such
    that
  
      C = A.dot(A.T).                                   (2)

    W is then just the inverse of A.  The decomposition of C into W
    is not unique, but since The solution for any least squares 
    problem depends on C, rather than W, any value of W should do 
    just fine as long as it satisfies (1).
  
  Notes
  -----
    This function can be slow

  '''
  C = np.asarray(C)
  
  # if C is an empty array then return an empty array 
  if C.size == 0:
    return np.copy(C)

  if is1d(C):
    W = 1.0/np.sqrt(C)

  else:
    N = np.shape(C)[0]
    A = scipy.linalg.cholesky(C,lower=True)
    W = scipy.linalg.solve_triangular(A,np.eye(N),lower=True,overwrite_b=True)

  return W
  
##------------------------------------------------------------------------------
def _residual(system,
               data,
               data_weight,
               prior,
               system_args,
               system_kwargs,
               jacobian,
               jacobian_args,
               jacobian_kwargs,
               reg_matrix,
               lm_matrix,
               bayes_matrix,
               data_indices):
  '''
  used for nonlin_lstsq
  '''  
  def residual_function(model):
    '''
    evaluates the function to be minimized for the given model

    '''
    pred = system(model,*system_args,**system_kwargs)
    res = pred - data
    res = res[data_indices]
    if is1d(data_weight):
      res = data_weight*res
    else:
      res = data_weight.dot(res)

    reg = reg_matrix.dot(model)    
    bayes = bayes_matrix.dot(model - prior)
    lm = lm_matrix.dot(0*model)
    return np.hstack((lm,res,reg,bayes))

  def residual_jacobian(model):
    '''
    evaluates the jacobian of the objective function at the given model

    The Jacobian consists of:

     |    df/dm   |
     |    L_r     |
     |    L_l     |
     |    L_b     |
    
    where
    
    df/dm is the derivative of the system with respect to the model
    parameters evaluated at the current model 

    L_r is the regularization matrix

    L_l is an identity matrix or an empty array depending on whether
    Levenberg Marquardt damping is used

    L_b is an identity matrix or an empty array depending on whether 
    Bayesian least squares is used

    '''
    jac = jacobian(model,*jacobian_args,**jacobian_kwargs)
    jac = jac[data_indices,:]
    if is1d(data_weight):
      jac = data_weight[:,None]*jac
    else:
      jac = data_weight.dot(jac)

    jac = np.vstack((lm_matrix,jac,reg_matrix,bayes_matrix))
    return jac

  return residual_function,residual_jacobian

##------------------------------------------------------------------------------
def _arg_parser(args,kwargs):
  '''parses and checks arguments for nonlin_lstsq()'''
  assert len(args) == 3, 'nonlin_lstsq takes exactly 3 positional arguments'
  p = {'solver':solvers.lstsq,
       'LM_damping':True,
       'LM_param':1e-4,
       'LM_factor':2.0,
       'maxitr':50,
       'rtol':1.0e-4,
       'atol':1.0e-4,
       'data_covariance':None,
       'prior_covariance':None,
       'system_args':None,
       'system_kwargs':None,
       'jacobian':None,
       'jacobian_args':None,
       'jacobian_kwargs':None,
       'solver_args':None,
       'solver_kwargs':None,
       'data_indices':None,
       'regularization':None,
       'dtype':None,
       'output':None}

  for key,val in kwargs.items():
    assert key in p.keys(), (
      'invalid keyword argument for nonlin_lstsq: %s' % key)

  p.update(kwargs)
  p['system'] = args[0]
  p['data'] = np.asarray(args[1])
  p['m_k'] = args[2]

  # if the initial guess is an integer, then interpret it as length of the model
  # parameter vector and assume ones as the initial guess
  if type(p['m_k']) == int:
    p['m_k'] = np.ones(p['m_k'])

  p['m_k'] = np.asarray(p['m_k'])
  p['prior'] = np.copy(p['m_k'])


  # assert that only one type of prior uncertainty is given      
  if p['data_covariance'] is not None:
    p['data_covariance'] = np.asarray(p['data_covariance'])

  else:
    p['data_covariance'] = np.ones(len(p['data']))

  if p['prior_covariance'] is not None:
    p['prior_covariance'] = np.asarray(p['prior_covariance'])
    p['bayes_matrix'] = np.eye(len(p['m_k']))

  else:
    p['prior_covariance'] = np.zeros((0,0))
    p['bayes_matrix'] = np.zeros((0,len(p['m_k'])))

  if p['system_args'] is None:
    p['system_args'] = []

  if p['system_kwargs'] is None:
    p['system_kwargs'] = {}

  # if no jacobian is provided then set use the finite difference approximation
  if p['jacobian'] is None:
    p['jacobian'] = make_jacobian(p['system'])
    p['jacobian_args'] = p['system_args']
    p['jacobian_kwargs'] = p['system_kwargs']
  
  if p['jacobian_args'] is None:
    p['jacobian_args'] = []

  if p['jacobian_kwargs'] is None:
    p['jacobian_kwargs'] = {}

  if p['solver_args'] is None:
    p['solver_args'] = []

  if p['solver_kwargs'] is None:
    p['solver_kwargs'] = {}

  # default to assuming all data will be used.  This functionality is added 
  # to make cross validation easier
  if p['data_indices'] is None:
    p['data_indices'] = range(len(p['data']))

  # if regularization is a array or tuple of length 2 then assume it describes
  # the regularization order and the penalty parameter then create the
  # regularization matrix
  if np.shape(p['regularization'])==(2,):
    order = p['regularization'][0]
    mag = p['regularization'][1]
    p['regularization'] = mag*tikhonov_matrix(range(len(p['m_k'])),order)

  if p['regularization'] is None:
    p['regularization'] = np.zeros((0,len(p['m_k'])))

  # if regularization is given as a sparse matrix and unsparsify it
  #if hasattr(p['regularization'],'todense'):
  #  p['regularization'] = np.array(p['regularization'].todense())
 
  assert len(np.shape(p['regularization'])) == 2, (
    'regularization must be 2-D array or length 2 array')

  assert np.shape(p['regularization'])[1] == len(p['m_k']), (
    'second axis for the regularization matrix must have length equal to the '
    'number of model parameters')

  if p['LM_damping']:
    assert p['LM_param'] > 0.0,('Levenberg-Marquardt parameter must be greater '
                                'than 0.0')

    p['lm_matrix'] = p['LM_param']*np.eye(len(p['m_k']))

  else:
    p['lm_matrix'] = np.zeros((0,len(p['m_k'])))

  if p['output'] is None:
    p['output'] = ['solution'] 

  # cut down data_covariance
  if is1d(p['data_covariance']): 
    p['data_covariance'] = p['data_covariance'][p['data_indices']]

  else:
    data_no = len(p['data_indices'])
    idx1,idx2 = np.ix_(p['data_indices'],p['data_indices'])
    p['data_covariance'] = p['data_covariance'][idx1,idx2]
    # if there is any repeat index in data_indices set covariance to 0
    # This is done to make it possible to bootstrap 
    p['data_covariance'][(idx1==idx2) != np.eye(data_no,dtype=bool)] = 0

  p['data_weight'] = covariance_to_weight(p.pop('data_covariance'))
  p['prior_weight'] = covariance_to_weight(p.pop('prior_covariance'))

  if is1d(p['prior_weight']):
    p['bayes_matrix'] = p.pop('prior_weight')[:,None]*p['bayes_matrix']
  else:
    p['bayes_matrix'] = p.pop('prior_weight').dot(p['bayes_matrix'])

  return p

##------------------------------------------------------------------------------
def nonlin_lstsq(*args,**kwargs):
  '''Newtons method for solving a least squares problem

  PARAMETERS
  ----------
  *args 
  -----
 
   system: function where the first argument is a vector of model
      parameters and the remaining arguments are system args and
      system kwargs

    data: (N,) vector of data values

    prior: (M,) vector of model parameter initial guesses. 

                        OR

      an integer specifying the number of model parameters.  If an
      integer is provided then the initial guess is a vector of ones

  **kwargs 
  --------

    system_args: list of arguments to be passed to system following
      the model parameters (default None)

    system_kwargs: list of key word arguments to be passed to system
      following the model parameters (default None)

    jacobian: function which computes the jacobian w.r.t the model
      parameters. the first arguments is a vector of parameters and
      the remaining arguments are jacobian_args and jacobian_kwargs
      (default: inverse.jacobian_fd)

    jacobian_args: arguments to be passed to the jacobian function 
      (default: None)

    jacobian_kwargs: key word arguments to be passed to the jacobian
      function (default: None)

    solver: function which solves "G*m = d" for m, where the first two
      arguments are G and d.  inverse.lstsq, and inverse.nnls are
      wrappers for np.linalg.lstsq, and scipy.optimize.nnls and can be
      used here. Using nnls ensures that the output model parameters
      are non-negative.  inverse.bounded_lstsq can be used to bound
      the results of m.  The bounds must be specified using
      'solver_args'. See the documentation for inverse.bounded_lstsq
      for more details. (default: inverse.lstsq)

    solver_args: additional arguments for the solver after G and d
      (default: None)

    solver_kwargs: additional key word arguments for the solver 
      (default: None)

    data_covariance: data covariance matrix.  This can be any square
      array (sparse or dense) as long as it has the 'dot' method
      (default: np.eye(N))

    prior_covariance: prior covariance matrix.  This can any square
      array (sparse or dense) as long as it has the 'dot' method
      (default: np.eye(M))

    regularization: regularization matrix scaled by the penalty
      parameter.  This is a (*,M) array.

                        OR

      array of length 2 where the first argument is the tikhonov
      regularization order and the second argument is the penalty
      parameter.  The regularization matrix is assembled assuming that
      the position of the model parameters in the vector m corresponds
      to their spatial relationship. (default: None)

    LM_damping: flag indicating whether to use the Levenberg Marquart
      algorithm which damps step sizes in each iteration but ensures
      convergence (default: True)

    LM_param: starting value for the Levenberg Marquart parameter 
      (default: 1e-4)

    LM_factor: the levenberg-Marquart parameter is either multiplied
      or divided by this value depending on whether the algorithm is
      converging or diverging  (default: 2.0)

    maxitr: number of steps for the inversion (default: 50)

    rtol: Algorithm stops if relative L2 between successive iterations
      is below this value (default: 1e-4)

    atol: Algorithm stops if absolute L2 is below this value 
      (default: 1e-4)

    data_indices: indices of data that will be used in the
      inversion. (default: range(N))

    output: list of strings indicating what this function returns. Can
      be 'solution', 'solution_covariance', 'predicted',
      'predicted_covariance', 'misfit', or 'iterations' (default:
      ['solution'])

  Returns
  -------
    m_new: best fit model parameters

  Usage
  -----  
    In [1]: import numpy as np
    In [2]: from modest import nonlin_lstsq
    In [3]: def system(m,x):
     ...:     return m[0] + m[1]*m[0]*x
     ...: 
    In [4]: x = np.linspace(0,10,10)
    In [5]: m = np.array([1.0,2.0])
    In [6]: data = system(m,x)
    In [7]: nonlin_lstsq(system,data,2,system_args=(x,))
    Out[7]: array([ 1.,  2.])

  nonlin_lstsq also handles more difficult systems which can
  potentially yield undefined values for some values of m.  As long as
  'system' evaluated with the initial guess for m produces a finite
  output, then nonlin_lstsq will converge to a solution.

    In [8]: def system(m,x):
        return m[0]*np.log(m[1]*(x+1.0))
       ...: 

    In [9]: data = system(m,x)
    In [10]: nonlin_lstsq(system,data,[10.0,10.0],system_args=(x,))
    Out[10]: array([ 1.00000011,  1.99999989])

  what separates nonlin_lstsq from other nonlinear solvers is its
  ability to easily add regularization constraints to illposed
  problems.  The following further constrains the problem with a first
  order tikhonov regularization matrix scaled by a penalty parameter
  of 0.001.  The null space in this case is anywhere that the product
  of m[0] and m[1] is the same. The added regularization requires that
  m[0] = m[1].
 
    In [123]: def system(m,x):
        return m[0]*m[1]*x
       .....: 
    In [124]: data = system([2.0,5.0],x)
    In [125]: nonlin_lstsq(system2,data,2,system_args=(x,),regularization=(1,0.001))
    Out[157]: array([ 3.16227767,  3.16227767])

  regularization can also be added through the 'solver' argument.
  Here, the solution is bounded such that 2.0 = m[0] and 0 <= m[1] <=
  100.  These bounds are imposed by making the solver a bounded least
  squares solver and adding the two solver args, which are the minimum
  and maximum values for the model parameters

    In [25]: nonlin_lstsq(system,data,2,system_args=(x,),
                          solver=modest.bvls,
                          solver_args=([2.0,0.0],[2.0,100.0]))
    Out[25]: array([ 2., 5.])

  '''
  p = _arg_parser(args,kwargs)
  res_func,res_jac = _residual(p['system'],
                               p['data'],
                               p['data_weight'],
                               p['prior'],
                               p['system_args'],
                               p['system_kwargs'],
                               p['jacobian'],
                               p['jacobian_args'],
                               p['jacobian_kwargs'],
                               p['regularization'],
                               p['lm_matrix'],
                               p['bayes_matrix'],
                               p['data_indices'])
  J = res_jac(p['m_k'])
  d = res_func(p['m_k'])

  assert np.all(np.isfinite(J)), ('non-finite value encountered in the initial '
                                  'Jacobian matrix.  Try using a different '
                                  'initial guess for the model parameters')
  assert np.all(np.isfinite(d)), ('non-finite value encountered in the initial '
                                  'predicted data vector.  Try using a different '
                                  'initial guess for the model parameters')

  err_last = np.inf
  err_curr = np.linalg.norm(d)
  counter = 0 
  logger.debug('error at iteration %s: %s' % (counter,err_curr))    
  while True:
    if not np.isfinite(err_curr):
      logger.info('exited due to infinite error')
      break 

    if err_curr < p['atol']:
      logger.info('converged to atol: error %s' % err_curr)
      break

    if np.abs(err_last - err_curr) < p['rtol']:
      logger.info('converged to rtol: error %s' % err_curr)
      break

    if counter >= p['maxitr']:
      logger.info('finished due to maxitr: error %s' % err_curr)
      break

    m_new = p['solver'](J,
                        -d+J.dot(p['m_k']),
                        *p['solver_args'],
                        **p['solver_kwargs'])
    err_last = err_curr
    err_curr = np.linalg.norm(res_func(m_new))
    counter += 1

    logger.debug('error at iteration %s: %s' % (counter,err_curr))    
    if p['LM_damping']:
      if err_curr < err_last:
        p['lm_matrix'] /= p['LM_factor']
        p['LM_param'] /= p['LM_factor']
        logger.debug('decreasing LM parameter to %s' % p['LM_param'])

      else:
        while True:
          if err_curr < err_last:
            break 

          if np.abs(err_last - err_curr) < p['rtol']:
            break

          if counter >= p['maxitr']:
            break

          p['LM_param'] *= p['LM_factor']
          p['lm_matrix'] *= p['LM_factor']
          logger.debug('increasing LM parameter to %s' % p['LM_param'])
          J[:p['lm_matrix'].shape[0],:] *= p['LM_factor']
          m_new = p['solver'](J,
                              -d+J.dot(p['m_k']),
                              *p['solver_args'],
                              **p['solver_kwargs'])
          err_curr = np.linalg.norm(res_func(m_new))
          counter += 1
          logger.debug('error at iteration %s: %s' % (counter,err_curr))    

    p['m_k'] = m_new
    d = res_func(p['m_k'])
    J = res_jac(p['m_k'])

  output = ()
  for s in p['output']:
    if s == 'solution':
      output += p['m_k'],

    if s == 'solution_covariance':
      try:
        soln_cov = scipy.linalg.inv(J.T.dot(J))
      except np.linalg.linalg.LinAlgError:
        soln_cov = scipy.linalg.pinv(J.T.dot(J))

      output += soln_cov,

    if s == 'jacobian':
      output += J,

    if s == 'predicted':
      output += p['system'](p['m_k'],
                            *p['system_args'],
                            **p['system_kwargs']),

    if s == 'predicted_covariance':
      try:
        soln_cov = scipy.linalg.inv(J.T.dot(J))
      except np.linalg.linalg.LinAlgError:
        soln_cov = scipy.linalg.pinv(J.T.dot(J))

      obs_jac = p['jacobian'](p['m_k'],
                              *p['jacobian_args'],
                              **p['jacobian_kwargs'])
      output += obs_jac.dot(soln_cov).dot(obs_jac.T),

    if s == 'misfit':
      output += err_curr,

    if s == 'iterations':
      output += counter,

  if len(output) == 1:
    output = output[0]

  return output
