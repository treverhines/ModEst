#!/usr/bin/env python
from __future__ import division
import numpy as np
from nllstsq import jacobian_fd
from nllstsq import nonlin_lstsq
from converger import Converger
import logging
from timing import funtime
import timing
import os
import h5py
import scipy.linalg

logger = logging.getLogger(__name__)


def nonlin_lstsq_update(system,
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
                        mask=None,**kwargs):
  if mask is None:
    mask = np.zeros(len(data),dtype=bool)
  not_mask = ~mask   
  return nonlin_lstsq(system,
                      data,
                      prior,
                      jacobian=jacobian,
                      data_covariance=Cdata,
                      prior_covariance=Cprior,
                      system_args=system_args,
                      system_kwargs=system_kwargs,
                      jacobian_args=jacobian_args,
                      jacobian_kwargs=jacobian_kwargs,
                      rtol=rtol,
                      atol=atol,
                      maxitr=maxitr,
                      data_indices=np.nonzero(not_mask)[0],
                      output=['solution','solution_covariance'],
                      **kwargs)
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

  #def norm(r):
  #  return r.dot(np.linalg.inv(Cdata_m)).dot(r)     
  
  # note that the norm is not using the data covariance for the sake
  # of computational efficiency
  def norm(r):
    return r.dot(r)     

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
def adjust_temp_size(history,itr):
    pad_length = 1
    for k in history.keys():
      s = history[k].shape[0]
      if s <= itr:
        history[k].resize(itr+pad_length,0)


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
               solver=iekf_update,
               solver_kwargs=None, 
               state_rate_cov=None,
               temp_file='.temp.h5',
               core=True,
               light=False, 
               chunk_length=100):
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

    # check for valid name, this is useless if core=True
    if light:
      self.isopen = False

    if not light:
      d = 0
      hfile = temp_file 
      if core is True:    
        while True:
          try:
            temp = h5py.File(temp_file,'w-',driver='core',backing_store=False)
            break
          except IOError:
            temp_file = hfile + '_%s' % d
            d += 1        
      else:
        while True:
          try:
            temp = h5py.File(temp_file,'w-')
            break
          except IOError:
            temp_file = hfile + '_%s' % d
            d += 1        

      self.isopen = True
      temp.create_dataset('prior',
                          shape=(0,self.N),
                          maxshape=(None,self.N),
                          dtype=np.float64,
                          chunks=(chunk_length,self.N))

      temp.create_dataset('posterior',
                          shape=(0,self.N),
                          maxshape=(None,self.N),
                          dtype=np.float64,
                          chunks=(chunk_length,self.N))

      temp.create_dataset('smooth',
                          shape=(0,self.N),
                          maxshape=(None,self.N),
                          dtype=np.float64,
                          chunks=(chunk_length,self.N))

      temp.create_dataset('prior_covariance',
                          shape=(0,self.N,self.N),
                          maxshape=(None,self.N,self.N),
                          dtype=np.float64,
                          chunks=(chunk_length,self.N,self.N))

      temp.create_dataset('posterior_covariance',
                          shape=(0,self.N,self.N),
                          maxshape=(None,self.N,self.N),
                          dtype=np.float64,
                          chunks=(chunk_length,self.N,self.N))

      temp.create_dataset('smooth_covariance',
                          shape=(0,self.N,self.N),
                          maxshape=(None,self.N,self.N),
                          dtype=np.float64,
                          chunks=(chunk_length,self.N,self.N))

      temp.create_dataset('transition_jacobian',
                          shape=(0,self.N,self.N),
                          maxshape=(None,self.N,self.N),
                          dtype=np.float64,
                          chunks=(chunk_length,self.N,self.N))

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
    self.solver = solver
    self.solver_kwargs = solver_kwargs
    self.ojac_args = ojac_args
    self.ojac_kwargs = ojac_kwargs
    self.trans_args = trans_args
    self.trans_kwargs = trans_kwargs
    self.tjac_args = tjac_args
    self.tjac_kwargs = tjac_kwargs
    self.pcov_args = pcov_args
    self.pcov_kwargs = pcov_kwargs
    self.light = light
    if not self.light:
      self.history = temp
    self.itr = 0
    self.t = None

  @funtime
  def update(self,z,cov,t,mask=None):
    logging.info('updating prior with observations at time %s (iteration %s)' % (t,self.itr))
    out = self.solver(self.obs,self.ojac,z,
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

    if not self.light:
      adjust_temp_size(self.history,self.itr)
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

    if not self.light:
      adjust_temp_size(self.history,self.itr)
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

  def close(self):
    if self.isopen == True:
      filename = self.history.filename
      self.history.close()
      self.isopen = False    
      if os.path.exists(filename):
        os.remove(filename) 

  def __del__(self):
    self.close()

  @funtime
  def smooth(self):
    '''
    Smoothes the state variables using the Rauch-Tung-Striebel
    algorithm. This algorithm is intended for linear systems and may
    produce undesirable results if the forward problem is highly
    nonlinear
    '''
    if self.light:
      print('smooth method not available for when initiated with light mode')
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

