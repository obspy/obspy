#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.core installer

:copyright: The ObsPy Development Team (devs@obspy.org)
:license: GNU Lesser General Public License, Version 3 (LGPLv3)
"""

import distribute_setup
distribute_setup.use_setuptools()
from setuptools import find_packages, setup


VERSION = '0.2.2'


setup(
    name='obspy.core',
    version=VERSION,
    description="ObsPy core classes, Python for Seismological Observatories",
    long_description="""
    obspy.core - Core classes of ObsPy: Python for Seismological Observatories

    This class contains common methods and classes for ObsPy. It includes
    UTCDateTime, Stats, Stream and Trace classes and methods for reading 
    seismograms.
    
    For more information visit http://www.obspy.org.
    """,
    url='http://www.obspy.org',
    author='The ObsPy Development Team',
    author_email='devs@obspy.org',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: ' + \
        'GNU Library or Lesser General Public License (LGPL)',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Geophysics',
    ],
    keywords=['ObsPy', 'seismology'],
    packages=find_packages(exclude=['distribute_setup']),
    namespace_packages=['obspy'],
    zip_safe=True,
    install_requires=[
        'numpy>1.0.0',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de" + \
        "/svn/obspy/obspy.core/trunk#egg=obspy.core-dev",
    include_package_data=True,
    test_suite="obspy.core.tests.suite",
)
