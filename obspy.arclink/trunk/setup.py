#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.arclink installer

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from setuptools import find_packages, setup
import os


VERSION = open(os.path.join("obspy", "arclink", "VERSION.txt")).read()


setup(
    name='obspy.arclink',
    version=VERSION,
    description="ArcLink/WebDC request client for of ObsPy.",
    long_description="""
    obspy.arclink - ArcLink/WebDC request client for of ObsPy

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
    keywords=['ObsPy', 'Seismology', 'ArcLink', 'MiniSEED', 'SEED',
              'Inventory', 'Waveform'],
    packages=find_packages(),
    namespace_packages=['obspy'],
    zip_safe=False,
    install_requires=[
        'setuptools',
        'obspy.mseed',
        'lxml',
    ],
    download_url="https://svn.obspy.org" + \
        "/obspy.arclink/trunk#egg=obspy.arclink-dev",
    include_package_data=True,
    test_suite="obspy.arclink.tests.suite",
)
