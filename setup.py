import os
from pyfo import __version__
from setuptools import setup, find_packages


setup(
    name='pyfo',
    version=__version__,
    url='http://localhost',
    author='Matthias Vogelgesang',
    author_email='matthias.vogelgesang@kit.edu',
    packages=find_packages(exclude=['*.tests']),
    description="Python-to-OpenCL source compiler",
    long_description="Python-to-OpenCL source compiler",
)
