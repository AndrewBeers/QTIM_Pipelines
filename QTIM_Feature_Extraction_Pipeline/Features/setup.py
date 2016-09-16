try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Build import cythonize
import numpy

setup(
    name = "QTIM_Feature_Pipeline",
    ext_modules = cythonize('c_utils.pyx'),
    include_dirs = [numpy.get_include()]  # accepts a glob pattern
)