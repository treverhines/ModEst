#!/usr/bin/env python
from __future__ import division
import numpy as np
from nllstsq import jacobian_fd
from nllstsq import nonlin_lstsq
from converger import Converger
import logging
from timing import funtime
import timing
import h5py
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
                rtol=1e-6,
                atol=1e-6,
                maxitr=10,
                mask=None):
  '''
  Update function for Iterated Extended Kalman Filter.  This
  algorithm comes from [1].

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

  if mask is None:
    mask = np.zeros(len(data),dtype=bool)

  data_m = np.asarray(data[~mask])
  Cdata_m = np.asarray(Cdata[np.ix_(~mask,~mask)])
  prior = np.asarray(prior)
  Cprior = np.asarray(Cprior)
  eta = np.copy(prior)
  
  H = jacobian(eta,*jacobian_args,**jacobian_kwargs)
  H_m = H[~mask,:]

  K = Cprior.dot(H_m.transpose()).dot(
        np.linalg.inv(
          H_m.dot(Cprior).dot(H_m.transpose()) + Cdata_m))

  pred = system(eta,*system_args,**system_kwargs)
  pred_m = pred[~mask]

  res_m = data_m - pred_m

  def norm(r):
    return r.dot(np.linalg.inv(Cdata_m)).dot(r)     

  conv = Converger(atol,rtol,maxitr,norm=norm)
  status,message = conv.check(res_m,set_residual=True)
  logger.debug('initial guess ' + message)

  while not ((status == 0) | (status == 3)):
    eta = prior + K.dot(res_m - H_m.dot(prior - eta))
    pred = system(eta,*system_args,**system_kwargs)
    pred_m = pred[~mask]
    res_m = data_m - pred_m
    
    status,message = conv.check(res_m,set_residual=True)
    logger.debug(message)

    H = jacobian(eta,
                 *jacobian_args,
                 **jacobian_kwargs)
    H_m = H[~mask,:]

    K = Cprior.dot(H_m.transpose()).dot(
          np.linalg.inv(
            H_m.dot(Cprior).dot(H_m.transpose()) + Cdata_m))


  Ceta = (np.eye(len(eta)) - K.dot(H_m)).dot(Cprior)
  return eta,Ceta

@funtime
def pcov_numerical(tjac,state,dt,R,N=5,
                   tjac_args=None,
                   tjac_kwargs=None):
  '''
  Numerically computes the process covariance matrix

  This is done by taking the covariance of the state variable and then
  integrating it with the transition function over the specified time
  interval
  '''
  if tjac_args is None:
    tjac_args = ()
  if tjac_kwargs is None:
    tjac_kwargs = {}

  # spare myself the numerical integration if there is no stochastic
  # variable
  if np.all(R == 0):
    return R

  t = np.linspace(0,dt,N)
  d = (dt)/(N-1)
  J0 = tjac(state,t[0],*tjac_args,**tjac_kwargs)
  out = J0.dot(R).dot(J0.transpose())*d/2.0
  for ti in t[1:-1]:
    Ji = tjac(state,ti,*tjac_args,**tjac_kwargs)
    out += Ji.dot(R).dot(Ji.transpose())*d

  Jend = tjac(state,t[-1],*tjac_args,**tjac_kwargs)
  out += Jend.dot(R).dot(Jend.transpose())*d/2.0
  return out


def default_trans(state,dt):
  '''
  Defaut transition function, which simply returns the state variable
  '''
  return np.array(state,copy=True)


def make_default_tjac(trans):
  '''
  Creates the default transition jacobian function

  The transition jacobian function returns the jacobian matrix of the
  transition function with respect to the state parameters
  '''
  def default_tjac(state,dt,*args,**kwargs):
    return jacobian_fd(state,trans,
                       system_args=(dt,)+args,
                       system_kwargs=kwargs)
  return default_tjac


def make_default_pcov(tjac,state_rate_cov):
  '''
  Creates the default process covariance function

  The process covariance function returns a covariance matrix
  describing how uncertainty in stochastic state parameters propagates
  to the next time step through the transition function
  '''
  def default_pcov(state,dt,*args,**kwargs):
    return pcov_numerical(tjac,state,dt,
                          state_rate_cov,
                          tjac_args=args,
                          tjac_kwargs=kwargs)
  return default_pcov   


def make_default_ojac(obs):
  '''
  Creates the default observation jacobian function

  The observation jacobian function returns the jacobian matrix of the
  observation function with respect to the state parameters
  '''
  def default_ojac(state,t,*args,**kwargs):
    return jacobian_fd(state,obs,
                       system_args=(t,)+args,
                       system_kwargs=kwargs)
  return default_ojac


@funtime  
def adjust_history_size(history,itr):
    pad_length = 1
    for k in history.keys():
      s = history[k].shape[0]
      if s <= itr:
        if type(history) is h5py.File:
          history[k].resize(itr+pad_length,0)
        else:
          pad_shape = list(history[k].shape)
          pad_shape[0] = (itr - s) + pad_length
          pad = np.zeros(pad_shape)
          history[k] = np.concatenate((history[k],pad),0)

class KalmanFilter:
  def __init__(self,
               prior,
               prior_cov,
               obs,
               obs_args=None,
               obs_kwargs=None,
               trans=None,
               trans_args=None,
               trans_kwargs=None,
               tjac=None,
               tjac_args=None,
               tjac_kwargs=None,
               pcov=None,
               pcov_args=None,
               pcov_kwargs=None,
               ojac=None,  
               ojac_args=None,
               ojac_kwargs=None,
               solver_kwargs=None, 
               state_rate_cov=None,
               history_file=None,
               light=False):
    '''
    data = obs(state,t,*obs_args,**obs_kwargs)
    obs_jacobian = ojac(state,t,*ojac_args,**ojac_kwargs)
    
    new_state = trans(state,dt,*tjac_args,**tjac_kwargs)
    trans_jacobian = tjac(state,dt,*tjac_args,**tjac_kwargs)

    process_cov = pcov(state,dt,stat_rate_cov,*pcov_args,**pcov_kwargs) 
    '''
    self.N = len(prior)
    self.state = {'prior':prior,
                  'prior_covariance':prior_cov,
                  'posterior':None,
                  'posterior_covariance':None,
                  'smooth':None,
                  'smooth_covariance':None}

    if state_rate_cov is None:
      state_rate_cov = np.zeros((self.N,self.N))

    if trans is None:
      trans = default_trans

    if tjac is None:
      tjac = make_default_tjac(trans)
      if tjac_args is None:
        tjac_args = trans_args
      if tjac_kwargs is None:
        tjac_kwargs = trans_kwargs

    if pcov is None:
      pcov = make_default_pcov(tjac,state_rate_cov)
      if pcov_args is None:
        pcov_args = tjac_args
      if pcov_kwargs is None:
        pcov_kwargs = pcov_kwargs

    if ojac is None:
      ojac = make_default_ojac(obs)
      if ojac_args is None:
        ojac_args = obs_args
      if ojac_kwargs is None:
        ojac_kwargs = obs_kwargs

    if (history_file is None) & (light == False):
      # history is stored in RAM, which could potentially use many GB
      history = {'prior':np.zeros((0,self.N)),
                 'posterior':np.zeros((0,self.N)),
                 'smooth':np.zeros((0,self.N)),
                 'prior_covariance':np.zeros((0,self.N,self.N)),
                 'posterior_covariance':np.zeros((0,self.N,self.N)),
                 'smooth_covariance':np.zeros((0,self.N,self.N)),
                 'transition_jacobian':np.zeros((0,self.N,self.N))}

    elif (history_file is not None) & (light == False):
      history = h5py.File(history_file,'w')
      history.create_dataset('prior',
                             shape=(0,self.N),
                             maxshape=(None,self.N),
                             dtype=np.float64,
                             chunks=True)

      history.create_dataset('posterior',
                             shape=(0,self.N),
                             maxshape=(None,self.N),
                             dtype=np.float64,
                             chunks=True)

      history.create_dataset('smooth',
                             shape=(0,self.N),
                             maxshape=(None,self.N),
                             dtype=np.float64,
                             chunks=True)
      history.create_dataset('prior_covariance',
                             shape=(0,self.N,self.N),
                             maxshape=(None,self.N,self.N),
                             dtype=np.float64,
                             chunks=True)
      history.create_dataset('posterior_covariance',
                             shape=(0,self.N,self.N),
                             maxshape=(None,self.N,self.N),
                             chunks=True)
      history.create_dataset('smooth_covariance',
                             shape=(0,self.N,self.N),
                             maxshape=(None,self.N,self.N),
                             dtype=np.float64,
                             chunks=True)
      history.create_dataset('transition_jacobian',
                             shape=(0,self.N,self.N),
                             maxshape=(None,self.N,self.N),
                             dtype=np.float64,
                             chunks=True)
    else:
      history = None

    if obs_args is None:
      obs_args = ()

    if obs_kwargs is None:
      obs_kwargs = {}

    if solver_kwargs is None:
      solver_kwargs = {}

    if ojac_args is None:
      ojac_args = ()

    if ojac_kwargs is None:
      ojac_kwargs = {}

    if trans_args is None:
      trans_args = ()

    if trans_kwargs is None:
      trans_kwargs = {}

    if tjac_args is None:
      tjac_args = ()

    if tjac_kwargs is None:
      tjac_kwargs = {}

    if pcov_args is None:
      pcov_args = ()

    if pcov_kwargs is None:
      pcov_kwargs = {}
     
    self.obs = obs
    self.trans = trans
    self.tjac = tjac
    self.pcov = pcov
    self.ojac = ojac
    self.obs_args = obs_args
    self.obs_kwargs = obs_kwargs
    self.solver_kwargs = solver_kwargs
    self.ojac_args = ojac_args
    self.ojac_kwargs = ojac_kwargs
    self.trans_args = trans_args
    self.trans_kwargs = trans_kwargs
    self.tjac_args = tjac_args
    self.tjac_kwargs = tjac_kwargs
    self.pcov_args = pcov_args
    self.pcov_kwargs = pcov_kwargs
    self.history = history
    self.itr = 0
    self.t = None
    self.light = light

  @funtime
  def update(self,z,cov,t,mask=None):
    logging.info('updating prior with observations at time %s (iteration %s)' % (t,self.itr))
    out = iekf_update(self.obs,self.ojac,z,
                      self.state['prior'],cov,
                      self.state['prior_covariance'],
                      system_args=(t,)+self.obs_args,
                      system_kwargs=self.obs_kwargs,
                      jacobian_args=(t,)+self.ojac_args,
                      jacobian_kwargs=self.ojac_kwargs,
                      mask=mask,
                      **self.solver_kwargs)

    self.state['posterior'] = out[0]
    self.state['posterior_covariance'] = out[1]

    if self.light == False:
      adjust_history_size(self.history,self.itr)
      self.history['prior'][self.itr,:] = self.state['prior']
      self.history['prior_covariance'][self.itr,:,:] = self.state['prior_covariance']

      self.history['posterior'][self.itr,:] = self.state['posterior']
      self.history['posterior_covariance'][self.itr,:,:] = self.state['posterior_covariance']
      
    self.itr += 1
    self.t = t


  @funtime
  def predict(self,dt):
    logging.info('predicting prior for time %s (iteration %s)' % (self.t+dt,self.itr))
    F = self.tjac(self.state['posterior'],
                  dt,
                  *self.tjac_args,
                  **self.tjac_kwargs)

    Q = self.pcov(self.state['posterior'],
                  dt,
                  *self.pcov_args,
                  **self.pcov_kwargs)

    self.state['prior'] = self.trans(self.state['posterior'],
                                     dt,
                                     *self.trans_args,
                                     **self.trans_kwargs)

    self.state['prior_covariance'] = F.dot(
                                     self.state['posterior_covariance']).dot(
                                     F.transpose()) + Q

    if self.light == False:
      adjust_history_size(self.history,self.itr)
      self.history['transition_jacobian'][self.itr,:,:] = F

  def next(self,data,data_cov,t,mask=None):
    if self.t is not None:
      self.predict(t - self.t)    

    self.update(data,data_cov,t,mask=mask)

  def get_transition_jacobian(self):
    return self.tjac(self.state['posterior'],
                     dt,
                     *self.tjac_args,
                     **self.tjac_kwargs)

  def get_prior(self):
    return self.state['prior'],self.state['prior_covariance']

  def get_posterior(self):
    return self.state['posterior'],self.state['posterior_covariance']

  def get_smooth(self):
    return self.state['smooth'],self.state['smooth_covariance']

  def filter(self,data,data_covariance,time,mask=None,smooth=False):
    if mask is None:
      mask = [None for i in range(len(time))]

    for i,d,dc,t,m in zip(range(len(time)),data,data_covariance,time,mask):
      self.next(d,dc,t,mask=m)

    if smooth is True:
      self.smooth()
      return self.get_smooth()    

    else:
      return self.get_posterior()    

  @funtime
  def smooth(self):
    '''
    Smoothes the state variables using the Rauch-Tung-Striebel
    algorithm. This algorithm is intended for linear systems and may
    produce undesirable results if the forward problem is highly
    nonlinear
    '''
    if self.light == True:
      logger.warning('Cannot use "smooth" method because light flag was raised '
                     'upon initializating KalmanFilter object')
      return

    n = self.itr
    h = self.history
    h['smooth'][n-1,:] = h['posterior'][n-1,:] 
    h['smooth_covariance'][n-1,:,:] = h['posterior_covariance'][n-1,:,:]
    for k in range(n-1)[::-1]:
      logging.info('creating smoothed state for iteration %s' % k)
      Ck = h['posterior_covariance'][k,:,:].dot(
            h['transition_jacobian'][k+1,:,:].transpose()).dot(
              np.linalg.inv(h['prior_covariance'][k+1,:,:]))
      h['smooth'][k,:] = (
        h['posterior'][k,:] + 
        Ck.dot(h['smooth'][k+1,:] - h['prior'][k+1,:]))
      h['smooth_covariance'][k,:,:] = (
        h['posterior_covariance'][k,:,:] + 
        Ck.dot(h['smooth_covariance'][k+1,:,:] - 
               h['prior_covariance'][k+1,:,:]).dot(Ck.transpose()))

class _KalmanFilter:
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

    

                    
    
  





  
