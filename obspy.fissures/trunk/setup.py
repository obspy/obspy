#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.fissures installer

@copyright: The ObsPy Development Team (devs@obspy.org)
@license: GNU Lesser General Public License, Version 3 (LGPLv3)
"""

from setuptools import find_packages, setup


VERSION = '0.2.2'


setup(
    name='obspy.fissures',
    version=VERSION,
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
    url='http://www.obspy.org',
    author='The ObsPy Development Team',
    author_email='devs@obspy.org',
    classifiers=[],
    keywords=['ObsPy', 'seismology', 'fissures', 'DHI', 'IRIS', 'CORBA'],
    license='LGPLv3',
    packages=find_packages(),
    namespace_packages=['obspy'],
    zip_safe=True,
    install_requires=[
        'setuptools',
        'obspy.mseed>0.2.1',
#        'omniORB',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de" + \
        "/svn/obspy/obspy.fissures/trunk#egg=obspy.fissures-dev",
    platforms=['any'],
    include_package_data=True,
    test_suite="obspy.fissures.tests.suite",
)
