#!/usr/bin/env python
import logging
from multiprocessing import Process, Queue
import mkl

def _f(f,q_in,q_out):
  while True:
    i,a = q_in.get()
    if i == 'DONE':
      break

    q_out.put((i,f(a)))

def parmap(f,args,Nprocs=None):
  '''  
  evaluates [f(a) for a in args] in parallel
  '''
  # make sure that lower level functions are not running in parallel
  starting_threads = mkl.get_max_threads()
  if Nprocs is None:
    Nprocs = starting_threads

  if Nprocs < 1:
    raise ValueError('number of worker processes must be 1 or greater')
    
  mkl.set_num_threads(1)

  # q_in has a max size of 1 so that bcast_args is not copied over to 
  # the next process until absolutely necessary
  q_in = Queue(1)
  q_out = Queue()
  procs = []
  for i in range(Nprocs):
    p = Process(target=_f,args=(f,q_in,q_out))
    # process is starting and waiting for something to be put on q_in
    p.start()

  count = 0
  for a in args:
    q_in.put((count,a))
    count += 1

  # indicate that nothing else will be added
  for i in range(Nprocs):
    q_in.put(('DONE',None))

  out = [q_out.get() for i in range(count)]

  # terminate all processes
  for p in procs:
    p.join()

  sorted_out = [x for i,x in sorted(out)]

  mkl.set_num_threads(starting_threads)
  return sorted_out

