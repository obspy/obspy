#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy installer

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
    
    or
    
    GNU General Public License (GPL)
"""

import distribute_setup
distribute_setup.use_setuptools()
from setuptools import find_packages, setup


VERSION = '0.3.3'


setup(
    name='obspy',
    version=VERSION,
    description="ObsPy, Python for Seismological Observatories",
    long_description="""
    ObsPy, Python for Seismological Observatories

    Capabilities include filtering, instrument simulation, unified read and
    write support for GSE2, MSEED, SAC, triggering, imaging, XML-SEED and
    there is experimental support for Arclink and SeisHub servers.

    Note currently only obspy.core, obspy.gse2, obspy.mseed, obspy.sac and
    obspy.imaging are automatically installed.

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
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    keywords=['ObsPy', 'seismology'],
    packages=find_packages(exclude=['distribute_setup']),
    namespace_packages=['obspy'],
    zip_safe=False,
    install_requires=[
        'obspy.core>0.3.2',
        'obspy.imaging>0.3.2',
        'obspy.gse2>0.3.2',
        'obspy.mseed>0.3.2',
        'obspy.sac>0.3.2',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de" + \
        "/svn/obspy/obspy/trunk#egg=obspy-dev",
    include_package_data=True,
    test_suite="obspy.core.testing.suite",
)
