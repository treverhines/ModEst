#!/usr/bin/env python
import time as timemod
import datetime
import os
import numpy as np
import logging
from functools import wraps
import collections

logger = logging.getLogger(__name__)


def listify(a):
  '''
  recursively convert an iterable into a list
  '''
  out = []
  for i in a:
    if hasattr(i,'__iter__'):
      out += [listify(i)]
    else:
      out += [i]

  return out


def list_flatten(lst):
  '''
  recursively flatten an iterable into a 1D list
  '''
  lst = list(lst)
  out = []
  for sub in lst:
    if hasattr(sub,'__iter__'):
      out.extend(list_flatten(sub))
    else:
      out.append(sub)
  return np.array(out)


def divide_list(lst,N):
  '''                               
  Splits a list into N groups as evenly as possible         
  '''
  if len(lst) < N:
    N = len(lst)
  out = [[] for i in range(N)]
  for itr,l in enumerate(lst):
    out[itr%N] += [l]
  return out


def find_indices(domain,realizations):
  '''  
  returns an array of indices such that domain[indices[n]] == realizations[n]
  '''
  if not hasattr(domain,'index'):
    domain = list(domain)

  if not hasattr(realizations,'index'):
    realizations = list(realizations)
  
  domain_set = set(domain)
  realizations_set = set(realizations)
  if len(domain_set) != len(domain):
      raise ValueError('domain contains repeated values')

  if not realizations_set.issubset(domain_set):
    intersection = realizations_set.intersection(domain_set)
    not_in_domain = realizations_set.difference(intersection)
    for i in not_in_domain:
      raise ValueError('item %s not found in domain' % i)

  indices = [domain.index(r) for r in realizations]
  return indices


def pad(M,pad_shape,value=0,dtype=None):
  '''
  returns an array containing the values from M but the ends of each dimension
  are padded with 'value' so that the returned array has shape 'pad_shape'
  '''
  M = np.array(M)
  M_shape = np.shape(M)

  assert len(M_shape) == len(pad_shape), ('new_shape must have the same '
         'number of dimensions as M') 
  assert all([m <= n for m,n in zip(M_shape,pad_shape)]), ('length of each new '
         'dimension must be greater than or equal to the corresponding '
         'dimension of M')

  if dtype is None:
    dtype = M.dtype

  out = np.empty(pad_shape,dtype=dtype)
  out[...] = value
 
  if not all(M_shape):
    return out

  M_dimension_ranges = [range(m) for m in M_shape]
  out[np.ix_(*M_dimension_ranges)] = M

  return out


def pad_stack(arrays,axis=0,**kwargs):
  '''
  stacks array along the specified dimensions and any inconsistent dimension
  sizes are dealt with by padding the smaller array with 'value'  
  '''
  array_shapes = [np.shape(a) for a in arrays]
  array_shapes = np.array(array_shapes)
  pad_shape = np.max(array_shapes,0)
  padded_arrays = []
  for a in arrays:
    a_shape = np.shape(a)
    pad_shape[axis] = a_shape[axis]     
    padded_arrays += [pad(a,pad_shape,**kwargs)]
  
  out = np.concatenate(padded_arrays,axis=axis)

  return out
    


      
