#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.seisan installer

@copyright: The ObsPy Development Team (devs@obspy.org)
@license: GNU Lesser General Public License, Version 3 (LGPLv3)
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
    classifiers=[],
    keywords=['ObsPy', 'seismology', 'SEISAN', 'waveform', 'seismograms'],
    license='LGPLv3',
    packages=find_packages(),
    namespace_packages=['obspy'],
    zip_safe=True,
    install_requires=[
        'setuptools',
        'obspy.core>0.2.1',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de" + \
        "/svn/obspy/obspy.seisan/trunk#egg=obspy.seisan-dev",
    platforms=['any'],
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
