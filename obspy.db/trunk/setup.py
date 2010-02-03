#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.db installer

:copyright: The ObsPy Development Team (devs@obspy.org)
:license: GNU Lesser General Public License, Version 3 (LGPLv3)
"""

from setuptools import setup, find_packages


VERSION = '0.0.1'


setup(
    name='obspy.db',
    version=VERSION,
    description="A waveform database for ObsPy.",
    long_description="""
    obspy.db - A waveform database for ObsPy.
    
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
    keywords=['ObsPy', 'seismology', 'seismogram', 'database', 'SeisHub'],
    packages=find_packages(),
    namespace_packages=['obspy'],
    zip_safe=True,
    install_requires=[
        'setuptools',
        'obspy.core>0.2.1',
        'obspy.mseed>0.2.1',
        'sqlalchemy',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de" + \
        "/svn/obspy/obspy.db/trunk#egg=obspy.db-dev",
    test_suite="obspy.sh.tests.suite",
    include_package_data=True,
    entry_points="""
        [obspy.db.feature]
        minmax_amplitude = obspy.db.feature:getMinMaxAmplitude
        mseed_timing_quality = obspy.db.feature:getTimingQualitity
    """,
)
