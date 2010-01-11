#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.seisan installer

:copyright: The ObsPy Development Team (devs@obspy.org)
:license: GNU Lesser General Public License, Version 3 (LGPLv3)
"""

from setuptools import find_packages, setup


VERSION = '0.2.0'


setup(
    name='obspy.seisan',
    version=VERSION,
    description="Read & write seismograms, Format SEISAN",
    long_description="""
    obspy.seisan - Read & write seismograms, Format SEISAN
    
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
    keywords=['ObsPy', 'seismology', 'SEISAN', 'waveform', 'seismograms'],
    packages=find_packages(),
    namespace_packages=['obspy'],
    zip_safe=True,
    install_requires=[
        'setuptools',
        'obspy.core>0.2.1',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de" + \
        "/svn/obspy/obspy.seisan/trunk#egg=obspy.seisan-dev",
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
