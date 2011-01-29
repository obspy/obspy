#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.iris installer

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from setuptools import find_packages, setup
import os


VERSION = open(os.path.join("obspy", "iris", "VERSION.txt")).read()


setup(
    name='obspy.iris',
    version=VERSION,
    description="Provides tools for accessing various IRIS web services.",
    long_description="""
    obspy.iris - Provides tools for accessing various IRIS web services.

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
    keywords=['ObsPy', 'seismology', 'IRIS', 'Waveform'],
    packages=find_packages(),
    namespace_packages=['obspy'],
    zip_safe=False,
    install_requires=[
        'setuptools',
        'obspy.core',
        'obspy.mseed',
        'lxml',
    ],
    download_url="https://svn.obspy.org/trunk/obspy.iris#egg=obspy.iris-dev",
    include_package_data=True,
    test_suite="obspy.iris.tests.suite",
    entry_points={},
)
