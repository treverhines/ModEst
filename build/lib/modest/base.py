#!/usr/bin/env python
import time as timemod
import datetime
import os
import numpy as np
import logging
from functools import wraps
import collections

logger = logging.getLogger(__name__)

def _baseN_to_base10(value_baseN,base_char):
  N = len(base_char)
  value_baseN = str(value_baseN)  
  base_char = base_char[:N]
  assert all(i in base_char for i in value_baseN), (
    'a character in the input does not exist in the base characters')

  value_base10 = sum(base_char.find(i)*N**(n) for (n,i) in enumerate(value_baseN[::-1]))
  return value_base10  


def _base10_to_baseN(value_base10,base_char):
  N = len(base_char)
  value_baseN = ""
  while value_base10 != 0:
    value_baseN = base_char[value_base10%N] + value_baseN
    value_base10 = value_base10//N
  if len(value_baseN) == 0:
    value_baseN = base_char[0]
  return value_baseN


def baseNtoM(value,N,M):
  '''
  converts an integer in base N to a value in base M

  PARAMETERS
  ----------
    value: (integer or string) the integer value in base N whose 
      characters must be a subset of the input base characters. If all the 
      characters in this value are 0-9 then an integer can be given, otherwise, 
      it must be a string.

    N: (integer or string): Specifies the input base characters. if N is an 
      integer then the base characters will be DEFAULT_CHARACTERS[:N]. Or the 
      base characters can be specified as a string. For example, specifying N 
      as '0123456789ABCDEF' will cause the input value to be treated as 
      hexidecimal. Alternatively, specifying N as 16 will cause the input to be 
      treated as hexidecimal.

    M: (integer or string): Base of the output value. 

  RETURNS
  -------
    value_baseM: a string of value_baseN converted to base M


  DEFAULT_CHARACTERS='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

  '''
  DEFAULT_CHARACTERS='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
  assert (type(N) == int) | (type(N) == str)
  assert (type(M) == int) | (type(M) == str)
  if type(N) is int:
    assert len(DEFAULT_CHARACTERS) >= N,(
      'integer values of N and M must be %s or smaller' %len(DEFAULT_CHARACTERS)) 
    baseN_char = DEFAULT_CHARACTERS[:N]
  else:
    baseN_char = N

  if type(M) is int:
    assert len(DEFAULT_CHARACTERS) >= M, (
      'integer values of N and M must be %s or smaller' %len(DEFAULT_CHARACTERS)) 
    baseM_char = DEFAULT_CHARACTERS[:M]
  else:  
    baseM_char = M

  assert len(baseN_char) > 1, (
    'There must be more than 1 base character')
  assert len(baseM_char) > 1, (
    'There must be more than 1 base character')

  value_base10 = _baseN_to_base10(value,baseN_char)
  value_baseM = _base10_to_baseN(value_base10,baseM_char)
  return value_baseM
  

def timestamp(factor=1.0):
  '''
  Description:
  Returns base 36 value of output from time.time() plus a character
  that identifies the computer this function was called from
  '''
  value_base10 = int(timemod.time()*factor)
  return baseNtoM(value_base10,10,36)
