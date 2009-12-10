#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.wav installer

@copyright: The ObsPy Development Team (devs@obspy.org)
@license: GNU General Public License (GPL)
    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 2
    of the License, or (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
    02110-1301, USA.
"""

from setuptools import setup, find_packages


VERSION = '0.2.0'

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
    classifiers=[],
    keywords=['ObsPy', 'seismology', 'seismogram', 'WAV'],
    license='GPL2',
    packages=find_packages(),
    namespace_packages=['obspy'],
    zip_safe=True,
    requires=[
        'setuptools',
        'obspy.core(>=0.2)',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de" + \
        "/svn/obspy/obspy.wav/trunk#egg=obspy.wav-dev",
    platforms=['any'],
    test_suite="obspy.wav.tests.suite",
    include_package_data=True,
)
