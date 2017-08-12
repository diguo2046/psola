#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
setup

Module installs psola package
Can be run via command: python setup.py install (or develop)

Author: jreinhold
Created on: Aug 09, 2017
"""

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

args = dict(
    name='psola',
    version='0.1.0',
    description="Time-Domain Pitch-Synchronous OverLap-and-Add algorithm",
    long_description=readme,
    author='Jacob Reinhold',
    author_email='jacob.reinhold@jhu.edu',
    url='https://github.com/jcreinhold/psola',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=["numpy>=1.13.1", "scipy>=0.19.1", "nose"],
    scripts=['psola/exec/psola_exec'],
    keywords="psola speech synthesis"
)

setup(**args)
