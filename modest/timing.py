#!/usr/bin/env python
import time as timemod
import datetime
import os
import numpy as np
import logging
from functools import wraps
import collections

logger = logging.getLogger(__name__)


def decyear_inv(decyear,format='%Y-%m-%dT%H:%M:%S'):
  '''
  convert decimal year to date
  '''
  year = int(np.floor(decyear))
  remainder = decyear - year
  year_start = datetime.datetime(year,1,1)
  year_end = datetime.datetime(year+1,1,1)
  days_in_year = (year_end - year_start).days
  decdays = remainder*days_in_year
  date = year_start + datetime.timedelta(days=decdays)
  return date.strftime(format)


def decyear(*args):
  '''
  converts date to decimal year
  '''
  date_tuple      = datetime.datetime(*args).timetuple()
  time_in_sec     = timemod.mktime(date_tuple)
  date_tuple      = datetime.datetime(args[0],1,1,0,0).timetuple()
  time_year_start = timemod.mktime(date_tuple)
  date_tuple      = datetime.datetime(args[0]+1,1,1,0,0).timetuple()
  time_year_end   = timemod.mktime(date_tuple)
  decimal_time    = (args[0] + (time_in_sec - time_year_start)
                     /(time_year_end - time_year_start))
  return decimal_time

class Timer:
  def __init__(self):
    self.time_dict = collections.OrderedDict()
    self.actime_dict = collections.OrderedDict()

  def tic(self,ID=None):
    if ID is None:
      itr = 0
      while True:
        n = 'process %s' % itr
        if not self.time_dict.has_key(n):
          ID = n
          break

        itr += 1

    if self.time_dict.has_key(ID):
      logger.warning('%s is already being timed' % ID)
      return 

    if not self.actime_dict.has_key(ID):
      self.actime_dict[ID] = 0.0

    self.time_dict[ID] = timemod.time()

    logger.debug('timing %s' % ID)

  def toc(self,ID=None):
    if ID is None:
      ID = self.time_dict.keys()[-1]

    curtime = timemod.time()
    runtime = curtime - self.time_dict.pop(ID)
    self.actime_dict[ID] += runtime

    disp = '%.4g %s' % self.convert(runtime)
    
    logger.debug('elapsed time for last call to %s: %s' % (ID,disp))
    return 'elapsed time for last call to %s: %s' % (ID,disp)

  def summary(self):
    while len(self.time_dict.keys()) > 0:
      self.toc()

    logger.info('---- TIME SUMMARY ----')
    for i,val in self.actime_dict.iteritems():    
      disp = '%.4g %s' % self.convert(val)
      logger.info('total time running %s: %s' % (i,disp))

  def convert(self,t):
    unit = 's'
    if t > 3600.0:
      unit = 'hr'
      t /= 3600.0

    elif t > 60.0:
      unit = 'min'
      t /= 60.0

    elif t < 1.0:
      unit = 'ms'
      t *= 1000.0

    return t,unit


GLOBAL_TIMER = Timer()


def funtime(fun):
  '''
  decorator which times a function
  '''
  @wraps(fun)
  def subfun(*args,**kwargs):
    t = Timer()
    GLOBAL_TIMER.tic(fun.__name__)
    out = fun(*args,**kwargs)
    GLOBAL_TIMER.toc(fun.__name__)
    return out
  return subfun


def tic(*args,**kwargs):
  return GLOBAL_TIMER.tic(*args,**kwargs)


def toc(*args,**kwargs):
  return GLOBAL_TIMER.toc(*args,**kwargs)


def summary(*args,**kwargs):
  return GLOBAL_TIMER.summary(*args,**kwargs)

