#!/usr/bin/env python
import _bvls

def bvls(G,d,bounds):
  '''
  Bounded Value Least Squares

  This is a python wrapper of the Fortran90 bvls module originally
  written by Charles Lawson and Richard Hanson and then modified by John
  Burkardt.

  USAGE
  -----
    >>>import bvls
    >>>G = np.random.random((10,2))
    >>>m = np.array([1.0,2.0])
    >>>d = G.dot(m)
    >>>lower_bound = np.array([0.0,0.0])
    >>>upper_bound = np.array([1.5,1.5])
    >>>output = bvls.bvls(G,d,[lower_bound,upper_bound])
    >>>m_est = output[0]
  '''
  return _bvls.bvls(G,d,bounds)

