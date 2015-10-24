#!/usr/bin/env python
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.add_extension('modest._bvls',sources=['modest/fortran/_bvls.pyf','modest/fortran/_bvls.f90'])
    return config

if __name__ == '__main__':
  from numpy.distutils.core import setup
  setup(name='ModEst',
        version='0.1',
        description='model estimation package for inverse problems',
        author='Trever Hines',
        author_email='treverhines@gmail.com',
        url='www.github.com/treverhines/ModEst',
        packages=['modest','modest/pymls'],
        license='MIT',
        configuration=configuration)

