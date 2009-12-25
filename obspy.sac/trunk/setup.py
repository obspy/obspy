#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.sac installer

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

from setuptools import find_packages, setup


VERSION = '0.2.2'


setup(
    name='obspy.sac',
    version=VERSION,
    description="Read & Write Seismograms, Format SAC.",
    long_description="""
    obspy.sac - Read & Write Seismograms, Format SAC.
    
    Python methods for reading and writing seismograms to SAC.

    For more information visit http://www.obspy.org.
    """,
    url='http://www.obspy.org',
    author='The ObsPy Development Team & C. J. Ammon',
    author_email='devs@obspy.org',
    classifiers=[],
    keywords=['ObsPy', 'seismology', 'SAC', 'waveform', 'seismograms'],
    license='GPL2',
    packages=find_packages(),
    namespace_packages=['obspy'],
    zip_safe=True,
    install_requires=[
        'setuptools',
        'obspy.core>0.2.1',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de" + \
        "/svn/obspy/obspy.sac/trunk#egg=obspy.sac-dev",
    platforms=['any'],
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
