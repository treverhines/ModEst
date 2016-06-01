#!/usr/bin/env python
# provides an easy interface with petsc
import numpy as np
import scipy.sparse
import logging
import modest
import matplotlib.pyplot as plt
try:
  import petsc4py
  petsc4py.init()
  from petsc4py import PETSc
except ImportError:
  print(
    'could not import PETSc. '
    'PETSc can be installed by following the instructions at '
    'https://www.mcs.anl.gov/petsc. Interfacing with PETSc requires '
    'petsc4py which can be found at https://bitbucket.org/petsc/petsc4py. '
    'Installing the latest version of petsc4py can be done with the command\n\n'
    '  pip install https://bitbucket.org/petsc/petsc4py/get/master.tar.gz\n')

  raise    

logger = logging.getLogger(__name__)


def _monitor(solver, its, fgnorm):
  ''' 
  this function is called for each iteration of a KSP solver
  '''
  logger.info('preconditioned residual norm at iteration %s: %.5e' % (its,fgnorm))


def petsc_solve(G,d,ksp='lgmres',pc='jacobi',rtol=1e-6,atol=1e-6,maxiter=1000,view=False):
  ''' 
  Solves a linear system using PETSc

  Parameters
  ----------
    G: (N,N) CSR sparse matrix
    d: (N,) data vector

    ksp: solve the system with this PETSc 
      routine. See PETSc documentation for a complete list of options.  
      'preonly' means that the system is solved with just the
      preconditioner. This is done when the preconditioner is 'lu', 
      which means that the system is directly solved with LU
      factorization. If the system is too large to allow for a direct
      solution then use an iterative solver such as 'lgmres' or 
      'gmres'

    pc: type of preconditioner. See PETSc documentation 
      for a complete list of options. 'jacobi' seems to work best for 
      iterative solvers. Use 'lu' if the solver is 'preonly'

    rtol: relative tolerance for iterative solvers
 
    atol: absolute tolerance for iterative solvers
  
    maxiter: maximum number of iterations

    view: logs information about the solver and monitors its 
      convergence

  '''
  converged_reason_lookup = {
    1:'KSP_CONVERGED_RTOL_NORMAL',
    9:'KSP_CONVERGED_ATOL_NORMAL',
    2:'KSP_CONVERGED_RTOL',
    3:'KSP_CONVERGED_ATOL',
    4:'KSP_CONVERGED_ITS',
    5:'KSP_CONVERGED_CG_NEG_CURVE',
    6:'KSP_CONVERGED_CG_CONSTRAINED',
    7:'KSP_CONVERGED_STEP_LENGTH',
    8:'KSP_CONVERGED_HAPPY_BREAKDOWN',
    -2:'KSP_DIVERGED_NULL',
    -3:'KSP_DIVERGED_ITS',
    -4:'KSP_DIVERGED_DTOL',
    -5:'KSP_DIVERGED_BREAKDOWN',
    -6:'KSP_DIVERGED_BREAKDOWN_BICG',
    -7:'KSP_DIVERGED_NONSYMMETRIC',
    -8:'KSP_DIVERGED_INDEFINITE_PC',
    -9:'KSP_DIVERGED_NANORINF',
    -10:'KSP_DIVERGED_INDEFINITE_MAT',
    -11:'KSP_DIVERGED_PCSETUP_FAILED',
    0:'KSP_CONVERGED_ITERATING'}

  if not scipy.sparse.isspmatrix(G):
    logger.info('system matrix is dense and will now be converted to a CSR sparse matrix')
    G = scipy.sparse.csr_matrix(G)

  #G += scipy.sparse.diags(1e-10*np.ones(G.shape[0]),0)
  G = G.tocsr() 

  #fig,ax = plt.subplots()
  #ax.imshow(G.toarray(),interpolation='none') 
  #plt.show()
  
  # instantiate LHS
  A = PETSc.Mat().createAIJ(size=G.shape,csr=(G.indptr,G.indices,G.data))

  # instantiate RHS
  d = PETSc.Vec().createWithArray(d)

  # create empty solution vector
  soln = np.zeros(G.shape[1])
  soln = PETSc.Vec().createWithArray(soln)

  # instantiate solver
  ksp_solver = PETSc.KSP()
  ksp_solver.create()
  ksp_solver.setType(ksp)
  ksp_solver.getPC().setType(pc)
  ksp_solver.setOperators(A)
  ksp_solver.setTolerances(rtol=rtol,atol=atol,max_it=maxiter)
  # solve and get information
  if view:  
    ksp_solver.view()
    ksp_solver.setMonitor(_monitor)

  ksp_solver.solve(d,soln)
  conv_number = ksp_solver.getConvergedReason()
  conv_reason = converged_reason_lookup[conv_number]
  if conv_number > 0:
    logger.debug('KSP solver converged due to %s' % conv_reason)
  else:
    logger.warning('KSP solver diverged due to %s' % conv_reason)
    print('WARNING: KSP solver diverged due to %s' % conv_reason)

  return soln.getArray()




