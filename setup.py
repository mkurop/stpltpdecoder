#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup

def read(fname):

    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "stpltpdecoder",
    version = "1.0.0",
    author = "Marcin Kuropatwiński",
    author_email = "marcin@talking2rabbit.com",
    description = "Short term/Long term predictors based speech encoder and decoder",
    license = "MIT",
    long_description = read('README.md'),
    py_modules = ['decoder', 'spectrum', 'utils'],
    )
