#!/usr/bin/env python
import numpy as np
import logging
import solvers
from converger import Converger
from tikhonov import Perturb
from tikhonov import tikhonov_matrix
from misc import funtime
import scipy.sparse

logger = logging.getLogger(__name__)

##------------------------------------------------------------------------------
@funtime
def jacobian_fd(m_o,
                system,
                system_args=None,
                system_kwargs=None,
                dm=1e-4,
                dtype=None):
  '''
  Parameters
  ----------

    system: function where the first argument is a list of model
      parameters and the output is a data list

    m_o: location in model space where the jacobian will be
      computed. must be a mutable sequence (e.g. np.array or list)

    system_args: additional arguments to system

    system_kargs: additional key word arguments to system

    dm: step size used for the finite difference approximation

  Returns
  -------
    J:  jacobian matrix with dimensions: len(data),len(parameters)

  ''' 
  if system_args is None:
    system_args = []
  if system_kwargs is None:
    system_kwargs = {}

  data_o = system(m_o,*system_args,**system_kwargs)
  data_o = np.asarray(data_o)
  param_no = len(m_o)
  data_no = len(data_o)
  Jac = np.zeros((data_no,param_no),dtype=dtype)
  i = 0
  for m_pert in Perturb(m_o,dm):
    data_pert = system(m_pert,*system_args,**system_kwargs)
    data_pert = np.asarray(data_pert)
    Jac[:,i]  = (data_pert - data_o)/dm
    i += 1

  return Jac

##------------------------------------------------------------------------------
@funtime
def covariance_to_weight(C):
  '''returns the weight matrix, W, which satisfies

       np.linalg.inv(C) = W.transpose().dot(W)                     (1)

     This is done using Cholesky factorization of C to find A such
     that
  
       C = A.dot(A.transpose()).                                   (2)

     W is then just the inverse of A.  The decomposition of C into W
     is not unique, but since The solution for any least squares 
     problem depends on C, rather than W, any value of W 'should' do 
     just fine as long as it satisfies (1).
  
     Notes
     -----

     This function is SLOW

  '''
  N = np.shape(C)[0]
  A = np.linalg.cholesky(C)
  W = scipy.linalg.solve_triangular(A,np.eye(N),lower=True)

  return W
  
##------------------------------------------------------------------------------
def _residual(system,
               data,
               data_weight,
               prior,
               prior_weight,
               system_args,
               system_kwargs,
               jacobian,
               jacobian_args,
               jacobian_kwargs,
               reg_matrix,
               lm_matrix,
               data_indices):

  '''
  used for nonlin_lstsq
  '''  
  @funtime
  def residual_function(model):
    '''
    evaluates the function to be minimized for the given model
    '''
    pred = system(model,*system_args,**system_kwargs)
    res = data_weight.dot(pred - data)
    res = res[data_indices]
    reg = reg_matrix.dot(model)    
    lm = np.zeros(np.shape(lm_matrix)[0])
    bayes = prior_weight.dot(model - prior)
    return np.hstack((res,reg,lm,bayes))

  @funtime
  def residual_jacobian(model):
    '''
    evaluates the jacobian of the objective function at the given model
    '''
    jac = jacobian(model,*jacobian_args,**jacobian_kwargs)
    jac = data_weight.dot(jac)
    jac = jac[data_indices,:]
    return np.vstack((jac,reg_matrix,lm_matrix,prior_weight))

  return residual_function,residual_jacobian

##------------------------------------------------------------------------------
@funtime
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
       'data_uncertainty':None,
       'data_covariance':None,
       'data_weight':None,
       'prior_uncertainty':None,
       'prior_covariance':None,
       'prior_weight':None,
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

  for key,val in kwargs.iteritems():
    assert key in p.keys(), (
      'invalid keyword argument for nonlin_lstsq(): %s' % key)

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
  assert np.sum([p['prior_uncertainty'] is not None,
                 p['prior_covariance'] is not None,
                 p['prior_weight'] is not None]) <= 1, (
  'Multiple types of prior uncertainty were specified')    

  assert np.sum([p['data_uncertainty'] is not None,
                 p['data_covariance'] is not None,
                 p['data_weight'] is not None]) <= 1, (
  'Multiple types of data uncertainty were specified')    

  # use whatever form of data uncertainty was provided to form the 
  # data weight matrix
  if p['data_weight'] is not None:
    p['data_weight'] = np.asarray(p['data_weight'])

  elif p['data_uncertainty'] is not None:
    if len(np.shape(p['data_uncertainty'])) == 1:
      sig = np.asarray(p['data_uncertainty'])
      p['data_weight'] = np.diag(1.0/sig)

    elif len(np.shape(p['data_uncertainty'])) == 2:
      cov = np.asarray(p['data_uncertainty'])
      p['data_weight'] = covariance_to_weight(cov)
      logger.warning(
        'use the data_covariance kwarg to specify covariance')

    else:
      raise ValueError(
        'data uncertainty must be 1 or 2 dimensional array')

  elif p['data_covariance'] is not None:
    cov = np.asarray(p['data_covariance'])
    p['data_weight'] = covariance_to_weight(cov)

  else:
    # if not data uncertainty was provided then weigh data with 
    # an identity matrix
    p['data_weight'] = np.eye(len(p['data']))

  assert np.sum([p['prior_uncertainty'] is not None,
                 p['prior_covariance'] is not None,
                 p['prior_weight'] is not None]) <= 1, (
  'Multiple types of data uncertainty were specified')    

  # use whatever form of prior uncertainty was provided to form 
  # the prior weight matrix
  if p['prior_weight'] is not None:
    p['prior_weight'] = np.asarray(p['prior_weight'])

  elif p['prior_uncertainty'] is not None:
    if len(np.shape(p['prior_uncertainty'])) == 1:
      sig = np.asarray(p['prior_uncertainty'])
      p['prior_weight'] = np.diag(1.0/sig)

    elif len(np.shape(p['prior_uncertainty'])) == 2:
      cov = np.asarray(p['prior_uncertainty'])
      p['prior_weight'] = covariance_to_weight(cov)
      logger.warning(
        'use the prior_covariance kwarg to specify covariance')

    else:
      raise ValueError(
        'data uncertainty must be 1 or 2 dimensional array')

  elif p['prior_covariance'] is not None:
    cov = np.asarray(p['prior_covariance'])
    p['prior_weight'] = covariance_to_weight(cov)

  else:
    # if no prior uncertainty is given, then the prior has no influence
    # on the final solution 
    p['prior_weight'] = np.zeros((0,len(p['m_k'])))

  if p['system_args'] is None:
    p['system_args'] = []

  if p['system_kwargs'] is None:
    p['system_kwargs'] = {}

  # if no jacobian is provided then set use the finite difference approximation
  if p['jacobian'] is None:
    p['jacobian'] = jacobian_fd
    p['jacobian_args'] = [p['system']]
    p['jacobian_kwargs'] = {'system_args':p['system_args'],
                            'system_kwargs':p['system_kwargs']}

  if p['jacobian_args'] is None:
    p['jacobian_args'] = []

  if p['jacobian_kwargs'] is None:
    p['jacobian_kwargs'] = {}

  if p['solver_args'] is None:
    p['solver_args'] = []

  if p['solver_kwargs'] is None:
    p['solver_kwargs'] = {}

  # default to assuming all data will be used.  This functionality is added for
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
  if hasattr(p['regularization'],'todense'):
    p['regularization'] = np.array(p['regularization'].todense())
 
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

  return p

##------------------------------------------------------------------------------
@funtime
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

    data_uncertainty: Vector of one standard deviation uncertainties
      for each data value.  This should be used if the data is assumed
      to be uncorrelated

                        OR

      data covariance matrix.  This can any square array (sparse or
      dense) as long as it has the 'dot' method (default: np.eye(N))

    prior_uncertainty: Vector of one standard deviation uncertainties
      for each prior value.  This should be used if the prior is assumed
      to be uncorrelated

                        OR

      prior covariance matrix.  This can any square array (sparse or
      dense) as long as it has the 'dot' method (default: np.eye(M))



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
      be 'solution', 'solution_uncertainty', 'predicted',
      'predicted_uncertainty', 'misfit', or 'iterations' (default:
      ['solution'])

  Returns
  -------
    m_new: best fit model parameters

  Usage
  -----  
    In [1]: import numpy as np
    In [2]: from inverse import nonlin_lstsq
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
                          solver=inverse.bounded_lstsq,
                          solver_args=([2.0,0.0],[2.0,100.0]))
    Out[25]: array([ 2., 5.])

  '''
  p = _arg_parser(args,kwargs)
 
  res_func,res_jac = _residual(p['system'],
                               p['data'],
                               p['data_weight'],
                               p['prior'],
                               p['prior_weight'],
                               p['system_args'],
                               p['system_kwargs'],
                               p['jacobian'],
                               p['jacobian_args'],
                               p['jacobian_kwargs'],
                               p['regularization'],
                               p['lm_matrix'],
                               p['data_indices'])

  final = np.zeros(len(res_func(p['m_k'])))

  #conv = Converger(final,atol=p['atol'],rtol=p['rtol'],maxitr=p['maxitr'])
  conv = Converger(atol=p['atol'],rtol=p['rtol'],maxitr=p['maxitr'])

  J = res_jac(p['m_k'])
  J = np.asarray(J,dtype=p['dtype'])
  d = res_func(p['m_k'])
  d = np.asarray(d,dtype=p['dtype'])

  assert np.all(np.isfinite(J)), ('non-finite value encountered in the initial '
                                  'Jacobian matrix.  Try using a different '
                                  'initial guess for the model parameters')
  assert np.all(np.isfinite(d)), ('non-finite value encountered in the initial '
                                  'predicted data vector.  Try using a different '
                                  'initial guess for the model parameters')

  status,message = conv.check(d)
  if status == 0:
    logger.info('initial guess ' + message)

  else:
    logger.debug('initial guess ' + message)

  while not ((status == 0) | (status == 3)):
    m_new = p['solver'](J,-d+J.dot(p['m_k']),
                        *p['solver_args'],
                        **p['solver_kwargs'])
    d_new = res_func(m_new)
    status,message = conv.check(d_new)
    if status == 0:
      logger.info(message)

    else:
      logger.debug(message)

    if (status == 1) and p['LM_damping']:
      logger.debug('decreasing LM parameter to %s' % p['LM_param'])
      p['lm_matrix'] /= p['LM_factor']
      p['LM_param'] /= p['LM_factor']

    while ((status == 2) | (status == 3)) and p['LM_damping']:
      logger.debug('increasing LM parameter to %s' % p['LM_param'])
      p['lm_matrix'] *= p['LM_factor']
      p['LM_param'] *= p['LM_factor']
      J = res_jac(p['m_k'])
      J = np.asarray(J,dtype=p['dtype'])
      d = res_func(p['m_k'])
      d = np.asarray(d,dtype=p['dtype'])
      m_new = p['solver'](J,-d+J.dot(p['m_k']),
                          *p['solver_args'],
                          **p['solver_kwargs'])
      d_new = res_func(m_new)
      status,message = conv.check(d_new)
      if status == 0:
        logger.info(message)
      else:
        logger.debug(message)
  
    p['m_k'] = m_new
    conv.set(d_new)

    J = res_jac(p['m_k'])
    J = np.asarray(J,dtype=p['dtype'])
    d = res_func(p['m_k'])
    d = np.asarray(d,dtype=p['dtype'])

  output = ()
  for s in p['output']:
    if s == 'solution':
      output += p['m_k'],

    if s == 'solution_uncertainty':
      output += np.linalg.inv(J.transpose().dot(J)),

    if s == 'jacobian':
      output += J,

    if s == 'predicted':
      output += p['system'](p['m_k'],
                            *p['system_args'],
                            **p['system_kwargs']),

    if s == 'predicted_uncertainty':
      soln_cov = np.linalg.inv(J.transpose().dot(J))
      obs_jac = p['jacobian'](p['m_k'],
                              *p['jacobian_args'],
                              **p['jacobian_kwargs'])
      output += obs_jac.dot(soln_cov).dot(obs_jac.transpose()),

    if s == 'misfit':
      output += conv.L2,

    if s == 'iterations':
      output += conv.itr,

  if len(output) == 1:
    output = output[0]

  return output
