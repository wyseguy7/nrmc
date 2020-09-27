import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import platform


if platform.system() == 'Windows':
    ext_modules = [] # windows is trash
else:
    ext_modules = [Extension("nrmc.biconnected",
                         ['src/nrmc/biconnected.pyx'],
                         language="c++",
                         extra_compile_args=['-march=native', '-O3', '-Wall', '-Wextra', '-pedantic',
                                            '-std=c++11', '-DFHT_HEADER_ONLY',  '-DMS_WIN64']
                         )]

# extra_compile_args=['-std=c++11'])]

setup(
    name="nrmc",
    version="0.0.1",
    author= "Evan Wyse",
    author_email="evan.wyse@duke.edu",
    description="Non-reversible Monte Carlo Markov Chain sampling over districting graphs",
    packages=setuptools.find_packages(where='src'), # unsure why this is necessary but it is
    ext_modules=ext_modules, cmdclass={"build_ext":build_ext},
    python_requires='>=3.6',
    # package_dir = {"": 'src/nrmc/src'}

    )

# TODO come up with a better package name
