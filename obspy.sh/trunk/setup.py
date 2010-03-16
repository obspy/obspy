#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.sh installer

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from setuptools import find_packages, setup
import os


VERSION = open(os.path.join("obspy", "sh", "VERSION.txt")).read()


setup(
    name='obspy.sh',
    version=VERSION,
    description="Support plug-in for Seismic Handler.",
    long_description="""
    obspy.sh - Support plug-in for Seismic Handler.
    
    This modules provides facilities to:
    - Import and export seismogram files in the Q format.
    - Import and export seismogram files in the ASC format.
    
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
    keywords=['ObsPy', 'seismology', 'seismogram', 'ASC', 'Q',
              'Seismic Handler'],
    packages=find_packages(),
    namespace_packages=['obspy'],
    zip_safe=False,
    install_requires=[
        'setuptools',
        'obspy.core',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de" + \
        "/svn/obspy/obspy.sh/trunk#egg=obspy.sh-dev",
    test_suite="obspy.sh.tests.suite",
    include_package_data=True,
    entry_points="""
        [obspy.plugin.waveform]
        Q = obspy.sh.core
        SH_ASC = obspy.sh.core

        [obspy.plugin.waveform.Q]
        isFormat = obspy.sh.core:isQ
        readFormat = obspy.sh.core:readQ
        writeFormat = obspy.sh.core:writeQ

        [obspy.plugin.waveform.SH_ASC]
        isFormat = obspy.sh.core:isASC
        readFormat = obspy.sh.core:readASC
        writeFormat = obspy.sh.core:writeASC
    """,
)
