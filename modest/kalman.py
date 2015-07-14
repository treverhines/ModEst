#!/usr/bin/env python
from __future__ import division
import numpy as np
from nllstsq import jacobian_fd
from nllstsq import nonlin_lstsq
from converger import Converger
import logging
from timing import funtime
import scipy.linalg

logger = logging.getLogger(__name__)

@funtime
def iekf_update(system,
                jacobian,
                data,
                prior,
                Cdata,
                Cprior,
                system_args=None,
                system_kwargs=None,
                jacobian_args=None,
                jacobian_kwargs=None,
                regularization=None,
                rtol=1e-6,
                atol=1e-6,
                maxitr=10):
  '''
  Update function for Iterated Extended Kalman Filter

  This algorithm comes from [1] and also allows for regularization by
  appropriately augmenting the observation function, observation
  Jacobian, and data.
 


  References
  ----------
  
    [1] Gibbs, P. B., "Advanced Kalman Filtering, Least-Squares and
          Modeling: A Practical Handbook". 2011, John Wiley & Sons, Inc

  '''
  if system_args is None:
    system_args = ()

  if system_kwargs is None:
    system_kwargs = {}

  if jacobian_args is None:
    jacobian_args = ()

  if jacobian_kwargs is None:
    jacobian_kwargs = {}

  if regularization is None:
    regularization = np.zeros((0,len(prior)))

  eta = np.copy(prior)
  reg = np.asarray(regularization)
  prior = np.asarray(prior)
  data = np.asarray(data)
  Cdata = scipy.linalg.block_diag(Cdata,
                                  np.eye(len(reg)))

  def norm(r):
    return r.dot(Cdata).dot(r)     

  conv = Converger(atol,rtol,maxitr,norm=norm)

  pred = system(eta,
                *system_args,
                **system_kwargs)
  res = data - pred
  res = np.hstack((res,
                   -reg.dot(eta)))

  status,message = conv.check(res,set_residual=True)
  if status == 0:
    logger.info('initial guess ' + message)

  else:
    logger.debug('initial guess ' + message)

  while not ((status == 0) | (status == 3)):
    H = jacobian(eta,
                 *jacobian_args,
                 **jacobian_kwargs)
    H = np.vstack((H,reg))
    K = Cprior.dot(H.transpose()).dot(
          np.linalg.inv(
            H.dot(Cprior).dot(H.transpose()) + Cdata))

    eta = prior + K.dot(res - H.dot(prior - eta))
    pred = system(eta,
                  *system_args,
                  **system_kwargs)
    res = data - pred
    res = np.hstack((res,
                     -reg.dot(eta)))
    status,message = conv.check(res,set_residual=True)
    if status == 0:
      logger.info(message)

    else:
      logger.debug(message)

  Ceta = (np.eye(len(eta)) - K.dot(H)).dot(Cprior)
  return eta,Ceta

class KalmanFilter2:
  def __init__(prior_mean,prior_covariance,transition,observation,time=None)
  def set_transition_arguments(self,args,kwargs):
  def set_update_arguments(self,args,kwargs):

class KalmanFilter:
  def __init__(self,prior,prior_cov,
               transition,
               observation,
               process_covariance,
               transition_jacobian=None,
               observation_jacobian=None):
    '''
    Parameters
    ----------
      
      prior: mean of the prior estimate of the state variable

      prior_cov: covariance of the prior estimate of the state variable

      transition: function which takes a posterior state as the first
        argument and returns the prior state for the next time step.
        This function is called in the 'predict' method.


      observation: function which takes a state variable as the first
        argument and returns the predicted obserables. This function
        is called in the 'update' method.


      process_covariance: function which returns the covariance of the
        process noise, which is the noise introduced by uncertainty in
        the transition function. This function is called in the
        'predict' method.

      transition_jacobian (optional): function which returns the
        derivative of the transition function with respect to the
        state variable evaluated at the provided state variable. This
        function is called in the 'predict' method. If not specified
        then a finite difference approximation of the jacobian is
        used.

      observation_jacobian (optional): function which returns the
        derivative of the observation function with respect to the
        state variable evaluated at the provided state variable. This
        function is called in the 'update' method. If this is not
        specified then a finite difference approximation of the
        jacobian is used.

    Methods
    -------
  
      get: retreive specified state variable

      predict: estimate prior for next time step

      update: compute posterior for current time step

    ''' 
    if transition_jacobian is None:
      transition_jacobian = jacobian_fd

    if observation_jacobian is None:
      observation_jacobian = jacobian_fd

    prior = np.asarray(prior)
    prior_cov = np.asarray(prior_cov)

    self.transition = transition
    self.transition_jacobian = transition_jacobian
    self.observation = observation
    self.observation_jacobian = observation_jacobian
    self.process_covariance = process_covariance
    self.new = {'prior':prior,
                'prior_covariance':prior_cov,
                'posterior':None,
                'posterior_covariance':None,
                'smooth':None,
                'smooth_covariance':None,
                'transition':None}
    self.state = []


  def _add_state(self):
    self.state += [self.new]
    self.new = {'prior':None,
                'prior_covariance':None,
                'posterior':None,
                'posterior_covariance':None,
                'smooth':None,
                'smooth_covariance':None,
                'transition':None}


  def get_prior(self):
    '''returns the prior for the current iteration'''
    if self.new['prior'] is None:
      raise ValueError(
        'prior has not yet been established for current iteration')

    return self.new['prior']


  def get_state_history(self,key):
    '''
    returns the specified type of state variable for each iteration

    Parameters 
    ---------- 

      key: can be either 'prior', 'prior_covariance', 'posterior', 'posterior_covariance',
        'smooth', 'smooth_covariance', or 'transition'

    '''
    return [i[key] for i in self.state]


  @funtime
  def predict(self,
              transition_args=None,
              transition_kwargs=None,
              jacobian_args=None,
              jacobian_kwargs=None,
              process_covariance_args=None,
              process_covariance_kwargs=None):
    '''
    estimates the prior state for the next iteration

    Parameters 
    ---------- 

      transition_args: additional arguments to be passed to the
        transition function after the state variable 

      transition_kwargs: additional key word arguments for the
        transition function 

      jacobian_args: additional arguments to be passed to
        transition_jacobian after the state variable. If
        transition_jacobian was not specified the the appropriate
        arguments for the finite difference approximation function are
        used.

      jacobian_kwargs: additional key word arguments to be passed to the
        transition_jacobian function

      process_covariance_args: arguments to be passed to
        process_covariance

      process_covariance_kwargs: key word arguments to be passed to
        process_covariance

    '''
    if transition_args is None:
      transition_args = ()

    if transition_kwargs is None:
      transition_kwargs = {}

    if jacobian_args is None:
      if self.transition_jacobian is jacobian_fd:
        jacobian_args = (self.transition,)
      else:
        jacobian_args = ()

    if jacobian_kwargs is None:
      if self.transition_jacobian is jacobian_fd:
        jacobian_kwargs = {'system_args':transition_args,
                           'system_kwargs':transition_kwargs}
  
      else:
        jacobian_kwargs = {}

    if process_covariance_args is None:
      process_covariance_args = ()

    if process_covariance_kwargs is None:
      process_covariance_kwargs = {}

    c = self.state[-1]
    assert c['posterior'] is not None

    F = self.transition_jacobian(c['posterior'],
                                 *jacobian_args,
                                 **jacobian_kwargs)

    Q = self.process_covariance(*process_covariance_args,
                                **process_covariance_kwargs)

    c['transition'] = F

    self.new['prior'] = self.transition(c['posterior'],
                                        *transition_args,
                                        **transition_kwargs)

    self.new['prior_covariance'] = F.dot(
                                   c['posterior_covariance']).dot(
                                   F.transpose()) + Q

  @funtime
  def update(self,z,R,
             observation_args=None,
             observation_kwargs=None,
             jacobian_args=None,
             jacobian_kwargs=None,
             solver_kwargs=None):
    '''
    uses a nonlinear Bayesian least squares algorithm with the given
    observables to update the prior for the current state

    Parameters
    ----------

      z: vector of observables
  
      R: covariance of observables
  
      observation_args: additional arguments to be passed to the
        observation function after the state variable

      observation_kwargs: additional key word arguments to be passed
        to the observation function after the state variable

      jacobian_args: additional arguments to be passed to
        observation_jacobian after the state variable. If
        observation_jacobian was not specified the the appropriate
        arguments for the finite difference approximation function are
        used.

      jacobian_kwargs: additional key word arguments to be passed to the
        observation_jacobian function
    
      solver_kwargs: additional key word arguments to be passed to
        inverse.nonlin_lstsq.

    '''
    if observation_args is None:
      observation_args = ()

    if observation_kwargs is None:
      observation_kwargs = {}

    if solver_kwargs is None:
      solver_kwargs = {}

    if jacobian_args is None:
      if self.observation_jacobian is jacobian_fd:
        jacobian_args = (self.observation,)
      else:
        jacobian_args = ()

    if jacobian_kwargs is None:
      if self.observation_jacobian is jacobian_fd:
        jacobian_kwargs ={'system_args':observation_args,
                          'system_kwargs':observation_kwargs}

      else:
        jacobian_kwargs = {}

    out = iekf_update(self.observation,
                self.observation_jacobian,
                z,
                self.new['prior'],
                R,
                self.new['prior_covariance'],
                system_args=observation_args,
                system_kwargs=observation_kwargs,
                jacobian_args=jacobian_args,
                jacobian_kwargs=jacobian_kwargs,
                **solver_kwargs)

    self.new['posterior'] = out[0]
    self.new['posterior_covariance'] = out[1]

    self._add_state()               

  @funtime
  def smooth(self):
    '''
    Smoothes the state variables using the Rauch-Tung-Striebel
    algorithm. This algorithm is intended for linear systems and may
    produce undesirable results if the forward problem is highly
    nonlinear
    '''
    N = len(self.state)
    clast = self.state[-1]
    clast['smooth'] = clast['posterior']
    clast['smooth_covariance'] = clast['posterior_covariance']
    for n in range(N-1)[::-1]:
      cnext = self.state[n+1]
      ccurr = self.state[n]
      C = ccurr['posterior_covariance'].dot(
          ccurr['transition'].transpose()).dot(
          np.linalg.inv(cnext['prior_covariance']))

      ccurr['smooth'] = (ccurr['posterior'] + 
                         C.dot( 
                         cnext['smooth'] -  
                         cnext['prior']))
      ccurr['smooth_covariance'] = (ccurr['posterior_covariance'] + 
                             C.dot(
                             cnext['smooth_covariance'] - 
                             cnext['prior_covariance']).dot(
                             C.transpose()))

    

                    
    
  





  
