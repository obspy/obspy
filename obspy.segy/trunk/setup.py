#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.segy installer

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from setuptools import find_packages, setup
import os


VERSION = open(os.path.join("obspy", "segy", "VERSION.txt")).read()


setup(
    name='obspy.segy',
    version=VERSION,
    description="Reading and writing support for the SEG Y format",
    long_description="""
    obspy.segy - Reading and writing support for the SEG Y format,
    
    This modules provides facilities to:
    - Import and export seismogram files in the SEG Y format
    
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
    keywords=['ObsPy', 'seismology', 'seismogram', 'SEG Y'],
    packages=find_packages(),
    namespace_packages=['obspy'],
    zip_safe=False,
    install_requires=[
        'setuptools',
        'obspy.core',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de" + \
        "/svn/obspy/obspy.segy/trunk#egg=obspy.segy-dev",
    test_suite="obspy.segy.tests.suite",
    include_package_data=True,
)
