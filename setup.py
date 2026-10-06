import numpy as np
from setuptools import setup, Extension

glcm = Extension('glcm',
                 sources=['py-glcm/core/src/glcm.cpp'],
                 include_dirs=[np.get_include()])

setup(name='py-glcm',
      version='1.1a0',
      description='py-glcm provides native implementations of GLCM related functions.',
      install_requires=['numpy>=1.19'],
      ext_modules=[glcm])
