  
from setuptools import find_packages, setup
import src

setup(
    name='src',
    packages=find_packages(),
    version=src.__version__,
    description='This repository contains a classification of Brazilian paper money.',
    author='Joao Victor Calvo Fracasso',
    license='BSD-3',
)