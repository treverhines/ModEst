#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import modest
import matplotlib.patches as patches

# define the nonlinear system which will be fit to data
def system(m,t):
  return m[0]*np.log(1 + t/m[1])

# bounds on model parameters
lb = np.array([1.4,0.15])
ub = np.array([1.7,0.25])


# number of model parameters
M = 2

# true mode we are trying to recover
model_true = np.random.random(2)
model_true[0] *= 0.6 
model_true[0] += 1.2 
model_true[1] *= 0.3
model_true[1] += 0.1
 
# create synthetic data
N = 100
time = np.linspace(0.0,5.0,100)
data = system(model_true,time) + np.random.normal(0.0,0.1,N)
data_variance = 0.1*np.ones(N)

soln,soln_cov,pred = modest.nonlin_lstsq(system,data,M,
                                         data_covariance=data_variance,
                                         system_args=(time,),
                                         solver=modest.bvls,
                                         solver_args=(lb,ub), 
                                         output=['solution','solution_covariance','predicted'])
# plot misfit surface
trialsx = np.linspace(1.0,2.0,200)
trialsy = np.linspace(0.001,0.5,200)
trialsx,trialsy = np.meshgrid(trialsx,trialsy)
trialsx = trialsx.flatten()
trialsy = trialsy.flatten()
trials = np.array([trialsx,trialsy]).T
misfit = np.array([np.sum((system(m,time) - data)**2/data_variance) for m in trials])
model_best = trials[np.argmin(misfit)]


# model space plot
fig,ax = plt.subplots()
plt.tripcolor(trials[:,0],trials[:,1],misfit,vmin=0.0,vmax=100.0,cmap='cubehelix')
plt.colorbar()
ax.errorbar(soln[0],soln[1],np.sqrt(soln_cov[1,1]),np.sqrt(soln_cov[0,0]),fmt='b.',markersize=15,elinewidth=3,capthick=3)
ax.plot(model_true[0],model_true[1],'ro',markersize=15)
ax.plot(model_best[0],model_best[1],'go',markersize=15)
ax.set_xlim(1.2,1.8)
ax.set_ylim(0.1,0.4)
ax.add_patch(patches.Rectangle(
               lb,
               ub[0]-lb[0],
               ub[1]-lb[1],
               fill=False))
 
# data space plot
plt.figure(2)
plt.errorbar(time,data,np.sqrt(data_variance),fmt='k.')
plt.plot(time,pred,'b-')
plt.show()



