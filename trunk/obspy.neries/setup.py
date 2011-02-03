#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.neries installer

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from setuptools import find_packages, setup
import os


VERSION = open(os.path.join("obspy", "neries", "VERSION.txt")).read()


setup(
    name='obspy.neries',
    version=VERSION,
    description="Provides tools for accessing various NERIES web services.",
    long_description="""
    obspy.neries - Provides tools for accessing various NERIES web services.

    For more information visit http://www.obspy.org.
    """,
    url='http://www.obspy.org',
    author='The ObsPy Development Team',
    author_email='devs@obspy.org',
    license='GNU Lesser General Public License, Version 3 (LGPLv3)',
    platforms='OS Independent',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: ' + \
        'GNU Library or Lesser General Public License (LGPL)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    keywords=['ObsPy', 'seismology', 'NERIES', 'Waveform', 'events', 'earthquakes'],
    packages=find_packages(),
    namespace_packages=['obspy'],
    zip_safe=False,
    install_requires=[
        'setuptools',
        'obspy.core',
        'lxml',
    ],
    download_url="https://svn.obspy.org/trunk/obspy.neries#egg=obspy.neries-dev",
    include_package_data=True,
    test_suite="obspy.neries.tests.suite",
    entry_points={},
)
