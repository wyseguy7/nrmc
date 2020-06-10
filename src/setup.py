from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

ext_modules = [Extension("biconnected", ['biconnected.pyx'], language="c++", extra_compile_args=['-std=c++11'])]


setup(ext_modules=ext_modules, cmdclass={"build_ext":build_ext})

# TODO we may need to