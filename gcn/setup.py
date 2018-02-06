from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext

MKLROOT="/opt/intel/compilers_and_libraries_2017.1.132/linux/mkl"

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("scheduler",
                           sources=["_scheduler.pyx", "scheduler.cpp", "mult.cpp"],
                           language="c++",
                           extra_compile_args=["-std=c++11", "-fopenmp"],
                           extra_link_args=['-lgomp'],
                           include_dirs=[numpy.get_include()]),
                 Extension("history",
                           sources=["_history.pyx", "history.cpp"],
                           language="c++",
                           extra_compile_args=["-std=c++11", "-fopenmp", "-march=native"],
                           extra_link_args=["-lpthread"],
                           include_dirs=[numpy.get_include()])
                ],
)
