#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.seisan installer

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from setuptools import find_packages, setup
import os


VERSION = open(os.path.join("obspy", "seisan", "VERSION.txt")).read()


setup(
    name='obspy.seisan',
    version=VERSION,
    description="Read seismograms, Format SEISAN",
    long_description="""
    obspy.seisan - Read seismograms, Format SEISAN
    
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
    keywords=['ObsPy', 'seismology', 'SEISAN', 'waveform', 'seismograms'],
    packages=find_packages(),
    namespace_packages=['obspy'],
    zip_safe=False,
    install_requires=[
        'setuptools',
        'obspy.core',
    ],
    download_url="https://svn.obspy.org" + \
        "/obspy.seisan/trunk#egg=obspy.seisan-dev",
    include_package_data=True,
    test_suite="obspy.seisan.tests.suite",
    entry_points="""
        [obspy.plugin.waveform]
        SEISAN = obspy.seisan.core

        [obspy.plugin.waveform.SEISAN]
        isFormat = obspy.seisan.core:isSEISAN
        readFormat = obspy.seisan.core:readSEISAN
    """,
)
