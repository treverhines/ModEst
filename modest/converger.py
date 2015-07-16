#!/usr/bin/env python
import numpy as np

class Converger:
  '''
  Class which checks residual and returns a status depending on how
  the residual is converging

  Status
  ------
    0: The residual being checked has converged due to some specified 
      tolerance, either atol, rtol or maxitr

    1: The residual being checked has a norm which is smaller than the
      currently set residual

    2: The residual being checked has a norm which is larger than the 
      currently set residual
   
    3: An invalid value (nan or inf) has been found in the residual 
      being checked

  Attributes
  ----------
    atol: 0 is raised when absolution error is below this value

    rtol: 0 is raised when relative error is below this value
 
    maxitr: total number of calls to to 'set_residual' (which should
      be the number of steps in model space) before 0 is
      raised. maxitr=0 will raise a convergence flag on the first
      call, maxitr=1 will raise it on the second call, etc.

    norm: callable which takes residual and returns a scalar

    error: norm(residual)

  '''
  def __init__(self,atol=1e-6,rtol=1e-6,maxitr=100,
               norm=np.linalg.norm):
    self.atol = atol
    self.rtol = rtol
    self.maxitr = maxitr
    self.norm = norm
    self.error = np.inf
    self.itr = 0

  def check(self,residual,set_residual=False):
    residual = np.asarray(residual)
    error_new = self.norm(residual)

    if self.itr >= self.maxitr:
      message = 'finished due to maxitr'
      out =  0,message

    elif not np.isfinite(error_new):
      message = 'encountered invalid L2'
      out = 3,message

    elif error_new <= self.atol:
      message = 'converged due to atol:          error=%s, itr=%s' % (error_new,self.itr)
      out = 0,message

    elif abs(error_new - self.error) <= self.rtol:
      message = 'converged due to rtol:          error=%s, itr=%s' % (error_new,self.itr)
      out = 0,message   

    elif error_new < self.error:
      message = 'converging:                     error=%s, itr=%s' % (error_new,self.itr)
      out = 1,message   

    elif error_new >= self.error:
      message = 'diverging:                      error=%s, itr=%s' % (error_new,self.itr)
      out =  2,message   

    if set_residual == True:
      self.set_residual(residual)

    return out
  
  def set_residual(self,residual):
    self.itr += 1
    residual = np.asarray(residual)
    self.error = self.norm(residual)
    return

    

