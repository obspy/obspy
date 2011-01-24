#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.core installer

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from setuptools import find_packages, setup
import distribute_setup
import os
distribute_setup.use_setuptools()


VERSION = open(os.path.join("obspy", "core", "VERSION.txt")).read()


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
    keywords=['ObsPy', 'seismology'],
    packages=find_packages(exclude=['distribute_setup']),
    namespace_packages=['obspy'],
    zip_safe=False,
    install_requires=[
        'numpy>1.0.0',
    ],
    download_url="https://svn.obspy.org/obspy.core/trunk#egg=obspy.core-dev",
    include_package_data=True,
    test_suite="obspy.core.tests.suite",
    entry_points={
        'console_scripts': [
            'obspy-runtests = obspy.core.scripts.runtests:main',
        ],
        'obspy.plugin.waveform': [
            'TSPAIR = obspy.core.ascii',
            'SLIST = obspy.core.ascii',
        ],
        'obspy.plugin.waveform.TSPAIR': [
            'isFormat = obspy.core.ascii:isTSPAIR',
            'readFormat = obspy.core.ascii:readTSPAIR',
        ],
        'obspy.plugin.waveform.SLIST': [
            'isFormat = obspy.core.ascii:isSLIST',
            'readFormat = obspy.core.ascii:readSLIST',
        ],
    },
)
