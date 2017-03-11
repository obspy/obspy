# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE.txt') as f:
    license = f.read()

setup(
    name='vcr',
    version='0.0.1',
    description='Decorator for capturing and simulating network communication',
    long_description=readme,
    author='The ObsPy Development Team',
    author_email='devs@obspy.org',
    url='https://github.com/obspy/vcr',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

