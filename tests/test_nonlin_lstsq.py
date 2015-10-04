#!/usr/bin/env python
import unittest
import modest
import numpy as np
tol = 1e-2
# make sure that nonlin_lstsq works for a linear case

def f_nonunique(model,x):
  out = 0.0*x
  for n,m in enumerate(model):
    out += m*np.sin(x/(1.0*(n+1)))*np.exp(-0.5*(n+1)*x)
  return out

def f(model,x):
  return model[0]*x + model[1]

def f_nonlin(model,x):
  return (model[0]*model[1])*x + model[1]

def jac(model,x):
  M = len(model)
  N = len(x)
  out = np.zeros((N,M))
  out[:,0] = x
  out[:,1] = 0.0*x + 1.0
  return out

class Test(unittest.TestCase):
  def test_linear(self):
    x = np.linspace(0,10,4)
    model_true = np.array([2.5,5.0])
    data = f(model_true,x)
    model_pred = modest.nonlin_lstsq(f,data,np.zeros(2),system_args=(x,))
    self.assertTrue(np.linalg.norm(model_pred - model_true) < tol)

  def test_linear_no_init_guess(self):
    x = np.linspace(0,10,4)
    model_true = np.array([2.5,5.0])
    data = f(model_true,x)
    model_pred = modest.nonlin_lstsq(f,data,2,system_args=(x,))
    self.assertTrue(np.linalg.norm(model_pred - model_true) < tol)

  def test_linear_with_jacobian(self):
    x = np.linspace(0,10,4)
    model_true = np.array([2.5,5.0])
    data = f(model_true,x)
    model_pred = modest.nonlin_lstsq(f,data,np.zeros(2),
                                      jacobian=jac,
                                      jacobian_args=(x,),
                                      system_args=(x,))
    self.assertTrue(np.linalg.norm(model_pred - model_true) < tol)

  def test_regularization(self):
    x = np.linspace(0,10,4)
    model_true = np.array([2.5,5.0])
    data = f(model_true,x)
    model_pred = modest.nonlin_lstsq(f,data,2,regularization=(0,0.0),
                                      system_args=(x,))
    self.assertTrue(np.linalg.norm(model_pred - model_true) < tol)
    model_pred = modest.nonlin_lstsq(f,data,2,regularization=(0,1e8),
                                      system_args=(x,))
    self.assertTrue(np.linalg.norm(model_pred) < tol)
    model_pred = modest.nonlin_lstsq(f,data,2,regularization=(1,1e8),
                                      system_args=(x,))
    self.assertTrue((model_pred[0] - model_pred[1]) < tol)

  def test_nonlinear(self):
    x = np.linspace(0,10,4)
    model_true = np.array([2.5,5.0])
    data = f_nonlin(model_true,x)
    model_pred = modest.nonlin_lstsq(f_nonlin,data,2,system_args=(x,))
    self.assertTrue(np.linalg.norm(model_pred - model_true) < tol)

  def test_solver_lstsq(self):
    x = np.linspace(0,10,4)
    model_true = np.array([2.5,5.0])
    data = f_nonlin(model_true,x)
    model_pred = modest.nonlin_lstsq(f_nonlin,data,2,system_args=(x,),solver=modest.lstsq)
    self.assertTrue(np.linalg.norm(model_pred - model_true) < tol)

  def test_solver_nnls(self):
    x = np.linspace(0,10,4)
    # the bounds should not impede the results here
    model_true = np.array([0.1,5.0])
    data = f_nonlin(model_true,x)
    model_pred = modest.nonlin_lstsq(f_nonlin,data,2,system_args=(x,),solver=modest.nnls)
    self.assertTrue(np.linalg.norm(model_pred - model_true) < tol)
    self.assertTrue(all(model_pred >= 0.0))

    # the bounds should influence the results here
    model_true = np.array([-2.5,5.0])
    data = f_nonlin(model_true,x)
    model_pred = modest.nonlin_lstsq(f_nonlin,data,2,system_args=(x,),solver=modest.nnls)
    self.assertTrue(all(model_pred >= 0.0))

  #def test_solver_bounded_lstsq(self):
  #  x = np.linspace(0,10)
  #  model_true = np.array([1.5,0.5])
  #  data = f(model_true,x)
  #  # the bounds should not impede the results here
  #  lower_bounds = np.array([1.0,0.0])
  #  upper_bounds = np.array([2.0,1.0])
  #  model_pred = modest.nonlin_lstsq(f,data,2,system_args=(x,),
  #                                   solver=modest.bounded_lstsq,
  #                                   solver_args=(lower_bounds,upper_bounds))
  #  print(model_pred)
  # 
  #  self.assertTrue(np.linalg.norm(model_pred - model_true) < tol)
  #  self.assertTrue(all(model_pred <= upper_bounds) & 
  #                  all(model_pred >= lower_bounds))

  # the bounds should influence the results here
  #  lower_bounds = np.array([-2.4,-2.4])
  #  upper_bounds = np.array([4.9,4.9])
  #  model_pred = modest.nonlin_lstsq(f_nonlin,data,2,system_args=(x,),
  #                                    solver=modest.bounded_lstsq,
  #                                    solver_args=(lower_bounds,upper_bounds))
  #
  #
  # self.assertTrue(all(model_pred <= upper_bounds) & 
  #                  all(model_pred >= lower_bounds))

unittest.main()

'''
print('checking modest.tikhonov for zeroth order')
reg = modest.tikhonov_matrix([0,1,2],0)
assert np.linalg.norm(reg - np.eye(3)) < tol
print('passed')

print('checking modest.tikhonov for first order')
reg = modest.tikhonov_matrix([0,1,2],1)
assert np.linalg.norm(reg - np.array([[-1.0,1.0,0.0],[0,-1.0,1.0]])) < tol, 'failed'
print('passed')

print('checking modest.tikhonov for second order')
reg = modest.tikhonov_matrix([0,1,2],2)
assert np.linalg.norm(reg - np.array([[-1.0,1.0,0.0],
                                      [1.0,-2.0,1.0],
                                      [0.0,1.0,-1.0]])) < tol, 'failed'
print('passed')

print('checking cross validation schemes in modest.penalty')
M = 10
N = 50
seed = 4
np.random.seed(seed)
x = np.linspace(0,10.0,N)
model_true = np.linspace(2.0,1.0,M)
data = f_nonunique(model_true,x)
sigma = 0.01*np.ones(N)
noise = np.random.normal(0,0.01,N)
data += noise
reg = modest.tikhonov_matrix(range(M),2)
penalty_range = np.power(10.0,np.linspace(-1,3,50))

pred = modest.GCV(penalty_range,f_nonunique,data,M,sigma=sigma,system_args=(x,),reg_matrix=reg)
best_pred_GCV = penalty_range[np.argmin(pred)]

pred = modest.CV(penalty_range,f_nonunique,data,M,sigma=sigma,system_args=(x,),reg_matrix=reg)
best_pred_CV = penalty_range[np.argmin(pred)]


pred = modest.KFCV(N,penalty_range,f_nonunique,data,M,sigma=sigma,system_args=(x,),reg_matrix=reg)
best_pred_KFCV = penalty_range[np.argmin(pred)]

assert np.abs(best_pred_GCV - best_pred_CV) < tol, 'failed'
assert np.abs(best_pred_CV - best_pred_KFCV) < tol, 'failed'
print('passed')
'''


