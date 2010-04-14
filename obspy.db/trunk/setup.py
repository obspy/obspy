#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.db installer

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from setuptools import find_packages, setup
import os


VERSION = open(os.path.join("obspy", "db", "VERSION.txt")).read()


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
    license='GNU Lesser General Public License, Version 3 (LGPLv3)',
    platforms='OS Independent',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: ' + \
        'GNU Library or Lesser General Public License (LGPL)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    keywords=['ObsPy', 'seismology', 'seismogram', 'database', 'SeisHub'],
    packages=find_packages(),
    namespace_packages=['obspy'],
    zip_safe=False,
    install_requires=[
        'setuptools',
        'obspy.core',
        'obspy.mseed',
        'sqlalchemy',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de" + \
        "/svn/obspy/obspy.db/trunk#egg=obspy.db-dev",
#    test_suite="obspy.db.tests.suite",
    entry_points={
        'obspy.db.feature' : [
            'minmax_amplitude = obspy.db.feature:MinMaxAmplitudeFeature',
        ],
        'console_scripts': [
            'obspy-indexer = obspy.db.scripts.indexer:main'
        ],
    }
)
