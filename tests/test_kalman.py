#!/usr/bin/env python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import modest
import scipy.linalg
import unittest
import logging
logging.basicConfig(level=logging.WARNING)

# Kalman filter test script
#
# The Kalman filter is identical to a bayesian least squares
# inversion, which can be performed by modest.nonlin_lstsq. This
# ensures that both modest.KalmanFilter and modest.nonlin_lstsq
# produce the same result for a line fitting example. I am finding the
# optimal values of a and b that satisfy data = a + b*time. The true
# answer should be a=2.0 and b=5.0

def system(state,time):
  return state[0] + state[1]*time

def system_kf(state,time):
  return np.array([state[0] + state[1]*time])

def system_reg_kf(state,time,reg):
  pred = np.array([state[0] + state[1]*time])
  reg_pred = reg.dot(state)
  return np.concatenate((pred,reg_pred))


time = np.array([ 0.,          0.35714286,  0.71428571,  
                  1.07142857,  1.42857143,  1.78571429,
                  2.14285714,  2.5,         2.85714286,  
                  3.21428571,  3.57142857,  3.92857143,
                  4.28571429,  4.64285714,  5.        ])

data = np.array([  2.,           3.7857143,    5.57142855,
                   7.35714285,   9.14285715,   10.92857145,  
                   12.7142857,   14.5,         16.2857143,
                   18.07142855,  19.85714285,  21.64285715,
                   23.42857145,  25.2142857,   27.        ])

data_kf = data[:,None]

data_reg_kf = np.concatenate((data_kf,np.zeros((len(data),2))),-1)

data_cov_kf = np.array([2.0*np.eye(1) for i in range(len(data))])


data_reg_cov_kf = np.array([scipy.linalg.block_diag(i,np.eye(2)) for i in data_cov_kf])

data_cov = scipy.linalg.block_diag(*data_cov_kf)

model_prior = np.array([3.0,3.0])
model_prior_cov = 5.0*np.eye(2)

reg_mat = np.eye(2)
reg_mat = np.array([[-1,1],[1,-1]])

class Test(unittest.TestCase):
  def test_bayes_least_squares(self):
    soln1,cov1 = modest.nonlin_lstsq(system,
                                data,
                                model_prior,
                                data_covariance=data_cov,
                                prior_covariance=model_prior_cov,
                                system_args=(time,),
                                output=['solution','solution_uncertainty'])
    
    kf = modest.KalmanFilter(model_prior,
                             model_prior_cov,
                             system_kf)

    kf.filter(data_kf,data_cov_kf,time)
    soln2,cov2 = kf.get_posterior()
    kf.close()
    self.assertTrue(np.all(np.isclose(soln1,soln2)))
    self.assertTrue(np.all(np.isclose(cov1,cov2)))

  def test_reg_bayes_least_squares(self):
    soln1 = modest.nonlin_lstsq(system,
                                data,
                                model_prior,
                                data_covariance=data_cov,
                                prior_covariance=model_prior_cov,
                                system_args=(time,),
                                regularization=reg_mat)
    kf = modest.KalmanFilter(model_prior,
                             model_prior_cov,
                             system_reg_kf,
                             obs_args=(0.25819889*reg_mat,))
    kf.filter(data_reg_kf,data_reg_cov_kf,time)
    soln2 = kf.get_posterior()[0]
    kf.close()
    self.assertTrue(np.all(np.isclose(soln1,soln2)))

  def test_masked_arrays(self):
    data_indices = [1,2,3,5,6,7,8,10,12]
    mask = np.ones((len(data),1),dtype=bool)
    mask[data_indices,:] = False
    soln1,cov1 = modest.nonlin_lstsq(system,
                                data,
                                model_prior,
                                data_covariance=data_cov,
                                prior_covariance=model_prior_cov,
                                data_indices=data_indices, 
                                system_args=(time,),
                                output=['solution','solution_uncertainty'])

    kf = modest.KalmanFilter(model_prior,
                             model_prior_cov,
                             system_kf)

    kf.filter(data_kf,data_cov_kf,time,mask=mask)
    soln2,cov2 = kf.get_posterior()
    kf.close()
    self.assertTrue(np.all(np.isclose(soln1,soln2)))
    self.assertTrue(np.all(np.isclose(cov1,cov2)))

  def test_smoothing_core(self):
    pred1 = modest.nonlin_lstsq(system,
                                data,
                                model_prior,
                                data_covariance=data_cov,
                                prior_covariance=model_prior_cov,
                                system_args=(time,),
                                output=['predicted'])

    kf = modest.KalmanFilter(model_prior,
                             model_prior_cov,
                             system_kf,
                             core=True)
 
    kf.filter(data_kf,data_cov_kf,time,smooth=True)
    pred2 = np.array([system_kf(kf.history['smooth'][i,:],t) for i,t in enumerate(time)])
    kf.close()
    self.assertTrue(np.all(np.isclose(pred1[:,None],pred2)))

  def test_smoothing_no_core(self):
    pred1 = modest.nonlin_lstsq(system,
                                data,
                                model_prior,
                                data_covariance=data_cov,
                                prior_covariance=model_prior_cov,
                                system_args=(time,),
                                output=['predicted'])

    kf = modest.KalmanFilter(model_prior,
                             model_prior_cov,
                             system_kf,
                             core=False)
 
    kf.filter(data_kf,data_cov_kf,time,smooth=True)
    pred2 = np.array([system_kf(kf.history['smooth'][i,:],t) for i,t in enumerate(time)])
    kf.close()
    self.assertTrue(np.all(np.isclose(pred1[:,None],pred2)))


unittest.main()
