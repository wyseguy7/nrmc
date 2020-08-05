from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import find_packages

ext_modules = [Extension("biconnected", ['biconnected.pyx'], language="c++", extra_compile_args=['-std=c++11'])]


setup(
    name="",
    version="0.0.1",
    author= "Evan Wyse",
    author_email="evan.wyse@duke.edu",
    description="Non-reversible MCMC chain sampling over districting graphs",
    packages=find_packages(),
    ext_modules=ext_modules, cmdclass={"build_ext":build_ext},
    python_requires='>=3.5',

    )

# TODO we may need to