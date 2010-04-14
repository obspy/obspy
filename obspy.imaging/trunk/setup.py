#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.imaging installer

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
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

from setuptools import find_packages, setup
import os


VERSION = open(os.path.join("obspy", "imaging", "VERSION.txt")).read()


setup(
    name='obspy.imaging',
    version=VERSION,
    description="Provides tools for displaying features used in seismology.",
    long_description="""
    obspy.imaging - Provides tools for displaying features used in seismology.

    For more information visit http://www.obspy.org.
    """,
    url='http://www.obspy.org',
    author='The ObsPy Development Team',
    author_email='devs@obspy.org',
    license='GNU General Public License (GPL)',
    platforms='OS Independent',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    keywords=['ObsPy', 'seismology', 'imaging', 'beachball',
              'focal mechanism', 'waveform', 'spectogram'],
    packages=find_packages(),
    namespace_packages=['obspy'],
    zip_safe=False,
    install_requires=[
        'setuptools',
        'obspy.core',
        'matplotlib',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de" + \
        "/svn/obspy/obspy.imaging/trunk#egg=obspy.imaging-dev",
    include_package_data=True,
    test_suite="obspy.imaging.tests.suite",
    entry_points={
        'console_scripts': [
            'obspy-scan = obspy.imaging.scripts.obspyscan:main',
            'obspy-plot = obspy.imaging.scripts.plot:main'
        ],
    },
)
