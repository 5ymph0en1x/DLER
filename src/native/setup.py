"""
Setup script for yenc_turbo native module
==========================================

Build with:
    python setup.py build_ext --inplace

Or install:
    pip install .
"""

from setuptools import setup, Extension
import sys
import os

# Extra compile args for maximum performance
extra_compile_args = []
extra_link_args = []

if sys.platform == 'win32':
    # MSVC or MinGW
    if 'MSC' in sys.version:
        extra_compile_args = ['/O2', '/GL', '/arch:AVX2']
        extra_link_args = ['/LTCG']
    else:
        # MinGW
        extra_compile_args = ['-O3', '-march=native', '-ffast-math']
        extra_link_args = ['-static-libgcc', '-static-libstdc++']
else:
    # Linux/Mac
    extra_compile_args = ['-O3', '-march=native', '-ffast-math', '-fPIC']

yenc_turbo = Extension(
    'yenc_turbo',
    sources=['yenc_turbo.cpp'],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language='c++',
)

setup(
    name='yenc_turbo',
    version='1.0.0',
    description='Ultra-fast yEnc decoder with GIL release',
    ext_modules=[yenc_turbo],
    python_requires='>=3.9',
)
