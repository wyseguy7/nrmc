import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

ext_modules = [Extension("nrmc/biconnected", ['biconnected.pyx'], language="c++", extra_compile_args=['-std=c++11'])]


setup(
    name="nrmc",
    version="0.0.1",
    author= "Evan Wyse",
    author_email="evan.wyse@duke.edu",
    description="Non-reversible Monte Carlo Markov Chain sampling over districting graphs",
    packages=setuptools.find_packages(),
    ext_modules=ext_modules, cmdclass={"build_ext":build_ext},
    python_requires='>=3.6',

    )

# TODO come up with a better package name