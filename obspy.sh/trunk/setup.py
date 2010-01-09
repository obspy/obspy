#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.sh installer

@copyright: The ObsPy Development Team (devs@obspy.org)
@license: GNU Lesser General Public License, Version 3 (LGPLv3)
"""

from setuptools import setup, find_packages


VERSION = '0.2.2'


setup(
    name='obspy.sh',
    version=VERSION,
    description="Support plug-in for Seismic Handler.",
    long_description="""
    obspy.sh - Support plug-in for Seismic Handler.
    
    This modules provides facilities to:
    - Import and export seismogram files in the Q format.
    - Import and export seismogram files in the ASC format.
    
    For more information visit http://www.obspy.org.
    """,
    url='http://www.obspy.org',
    author='The ObsPy Development Team',
    author_email='devs@obspy.org',
    classifiers=[],
    keywords=['ObsPy', 'seismology', 'seismogram', 'ASC', 'Q',
              'Seismic Handler'],
    license='LGPLv3',
    packages=find_packages(),
    namespace_packages=['obspy'],
    zip_safe=True,
    install_requires=[
        'setuptools',
        'obspy.core>0.2.1',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de" + \
        "/svn/obspy/obspy.sh/trunk#egg=obspy.sh-dev",
    platforms=['any'],
    test_suite="obspy.sh.tests.suite",
    include_package_data=True,
    entry_points="""
        [obspy.plugin.waveform]
        Q = obspy.sh.core
        SH-ASC = obspy.sh.core

        [obspy.plugin.waveform.Q]
        isFormat = obspy.sh.q:isQ
        readFormat = obspy.sh.q:readQ
        writeFormat = obspy.sh.q:writeQ

        [obspy.plugin.waveform.SH-ASC]
        isFormat = obspy.sh.asc:isASC
        readFormat = obspy.sh.asc:readASC
        writeFormat = obspy.sh.asc:writeASC
    """,
)
