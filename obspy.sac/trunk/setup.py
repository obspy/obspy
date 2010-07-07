#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.sac installer

:copyright:
    The ObsPy Development Team (devs@obspy.org) & C. J. Annon
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from setuptools import find_packages, setup
import os


VERSION = open(os.path.join("obspy", "sac", "VERSION.txt")).read()


setup(
    name='obspy.sac',
    version=VERSION,
    description="Read & Write Seismograms, Format SAC.",
    long_description="""
    obspy.sac - Read & Write Seismograms, Format SAC
    
    Python methods for reading and writing seismograms to SAC.

    For more information visit http://www.obspy.org.
    """,
    url='http://www.obspy.org',
    author='The ObsPy Development Team & C. J. Ammon',
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
    keywords=['ObsPy', 'seismology', 'SAC', 'waveform', 'seismograms'],
    packages=find_packages(),
    namespace_packages=['obspy'],
    zip_safe=False,
    install_requires=[
        'setuptools',
        'obspy.core',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de" + \
        "/svn/obspy/obspy.sac/trunk#egg=obspy.sac-dev",
    include_package_data=True,
    test_suite="obspy.sac.tests.suite",
    entry_points="""
        [obspy.plugin.waveform]
        SAC = obspy.sac.core

        [obspy.plugin.waveform.SAC]
        isFormat = obspy.sac.core:isSAC
        readFormat = obspy.sac.core:readSAC
        writeFormat = obspy.sac.core:writeSAC
    """,
)
