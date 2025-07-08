# Copyright (c) Facebook, Inc. and its affiliates.

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np


# hacky way to find numpy include path
# replace with actual path if this does not work
# np_include_path = include_dirs=[np.get_include()]
INCLUDE_PATH = [np.get_include()]

setup(
    ext_modules = cythonize(
            Extension(
                "box_intersection",
                sources=["box_intersection.pyx"],
                include_dirs=INCLUDE_PATH
            )),
)
