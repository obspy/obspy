#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.core installer

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


VERSION = '0.2.0'


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
    classifiers=[],
    keywords=['ObsPy', 'seismology'],
    license='GPL2',
    packages=find_packages(),
    namespace_packages=['obspy'],
    zip_safe=True,
    requires=[
        'setuptools',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de" + \
        "/svn/obspy/obspy.core/trunk#egg=obspy.core-dev",
    platforms=['any'],
    include_package_data=True,
    test_suite="obspy.core.tests.suite",
)
