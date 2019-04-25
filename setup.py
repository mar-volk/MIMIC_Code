#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io

from os.path import dirname
from os.path import join

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    file_path = join(dirname(__file__), *names)
    with io.open(file_path, encoding=kwargs.get('encoding', 'utf8')) as fh:
        return fh.read()


setup(
    name='MIMIC_Code',
    version='0.0.0',
    description=' - ',
    long_description=read('README.md'),
    author='Martin',
    author_email='martin_volk@gmx.net',
    url='https://github.com/mar-volk/medical_ml',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    zip_safe=False,
)
