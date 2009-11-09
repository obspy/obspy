# -*- coding: utf-8 -*-
"""
setup.py bdist_egg
"""

from setuptools import setup, find_packages

version = '0.0.1'

GPL2 = """
GNU General Public License (GPL)

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

setup(
    name='obspy.fissures',
    version=version,
    description="DHI/Fissures request client for of ObsPy.",
    long_description="""
    obspy.fissures - DHI/Fissures request client for of ObsPy.
    
    The Data Handling Interface (DHI) is a CORBA data access framework
    allowing users to access seismic data and metadata from IRIS DMC
    and other participating institutions directly from a DHI-supporting
    client program. The effect is to eliminate the extra steps of
    running separate query interfaces and downloading of data before
    visualization and processing can occur. The information is loaded
    directly into the application for immediate use.
    http://www.iris.edu/dhi/
    
    Detailed information on network_dc and seismogram_dc servers:
     * http://www.seis.sc.edu/wily
     * http://www.iris.edu/dhi/servers.htm
    """,
    classifiers=[],
    keywords='ObsPy, Seismology',
    author='The ObsPy Development Team',
    author_email='beyreuth@geophysik.uni-muenchen.de',
    url='https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.fissures',
    license=GPL2,
    packages=find_packages(exclude=['ez_setup']),
    namespace_packages=['obspy'],
    include_package_data=True,
    zip_safe=True,
    test_suite="obspy.fissures.tests.suite",
    install_requires=[
        'setuptools',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.fissures/trunk#egg=obspy.fissures-dev",
)
