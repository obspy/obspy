#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.core installer

:copyright: The ObsPy Development Team (devs@obspy.org)
:license: GNU Lesser General Public License, Version 3 (LGPLv3)
"""

import distribute_setup
distribute_setup.use_setuptools()
from setuptools import find_packages, setup


VERSION = '0.2.2'


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
    keywords=['ObsPy', 'seismology'],
    packages=find_packages(exclude=['distribute_setup']),
    namespace_packages=['obspy'],
    zip_safe=True,
    install_requires=[
        'obspy.core',
        'obspy.imaging',
        'obspy.gse2',
        'obspy.mseed',
        'obspy.sac',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de" + \
        "/svn/obspy/obspy/trunk#egg=obspy-dev",
    include_package_data=True,
    test_suite="obspy.core.testing.suite",
)
