from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext

MKLROOT="/opt/intel/compilers_and_libraries_2017.1.132/linux/mkl"

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("scheduler",
                           sources=["_scheduler.pyx", "scheduler.cpp"],
                           language="c++",
                           extra_compile_args=["-std=c++11", "-fopenmp"],
                           extra_link_args=['-lgomp'],
                           include_dirs=[numpy.get_include()]),
                 Extension("history",
                           sources=["_history.pyx", "history.cpp"],
                           language="c++",
                           extra_compile_args=["-std=c++11", "-fopenmp", "-march=native", "-m64", "-I${MKLROOT}/include"],
                           extra_link_args=["-Wl,--start-group", 
                               MKLROOT+"/lib/intel64/libmkl_intel_lp64.a", 
                               #MKLROOT+"/lib/intel64/libmkl_gnu_thread.a", 
                               #MKLROOT+"/lib/intel64/libmkl_sequential.a",
                               MKLROOT+"/lib/intel64/libmkl_intel_thread.a",
                               MKLROOT+"/lib/intel64/libmkl_core.a", "-Wl,--end-group", "-liomp5", "-lpthread", "-lm", "-ldl"],
                           include_dirs=[numpy.get_include()])
                ],
)
