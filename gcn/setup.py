from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("scheduler",
                 sources=["_scheduler.pyx", "scheduler.cpp"],
                 language="c++",
                 extra_compile_args=["-std=c++11", "-fopenmp"],
                 extra_link_args=['-lgomp'],
                 include_dirs=[numpy.get_include()])],
)
