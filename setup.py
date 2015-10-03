#!/usr/bin/env python
from numpy.distutils.core import setup
if __name__ == '__main__':
  setup(name='ModEst',
        version='0.1',
        description='model estimation package for inverse problems',
        author='Trever Hines',
        author_email='treverhines@gmail.com',
        url='www.github.com/treverhines/ModEst',
        packages=['modest','modest/pymls'],
        license='MIT',
        configuration=configuration)

