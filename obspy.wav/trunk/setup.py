#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.wav installer

:copyright: The ObsPy Development Team (devs@obspy.org)
:license: GNU Lesser General Public License, Version 3 (LGPLv3)
"""

from setuptools import setup, find_packages


VERSION = '0.2.2'


setup(
    name='obspy.wav',
    version=VERSION,
    description="Read & write seismograms, Format WAV.",
    long_description="""
    obspy.wav - Read & write seismograms, Format WAV.
    
    Python methods in order to read and write seismograms to WAV audio
    files. The data are squeezed to audible frequencies.
    
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
    keywords=['ObsPy', 'seismology', 'seismogram', 'WAV'],
    packages=find_packages(),
    namespace_packages=['obspy'],
    zip_safe=True,
    install_requires=[
        'setuptools',
        'obspy.core>0.2.1',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de" + \
        "/svn/obspy/obspy.wav/trunk#egg=obspy.wav-dev",
    test_suite="obspy.wav.tests.suite",
    include_package_data=True,
    entry_points="""
        [obspy.plugin.waveform]
        WAV = obspy.wav.core

        [obspy.plugin.waveform.WAV]
        isFormat = obspy.wav.core:isWAV
        readFormat = obspy.wav.core:readWAV
        writeFormat = obspy.wav.core:writeWAV
    """,
)
