#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.mseed installer

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
from setuptools.extension import Extension
import os
import sys


VERSION = '0.2.0'


# hack to prevent build_ext from trying to append "init" to the export symbols
class finallist(list):
    def append(self, object):
        return

class MyExtension(Extension):
    def __init__(self, *args, **kwargs):
        Extension.__init__(self, *args, **kwargs)
        self.export_symbols = finallist(self.export_symbols)

src = os.path.join('obspy', 'mseed', 'src', '2.3', 'libmseed') + os.sep
symbols = open(src + 'libmseed.def', 'r').readlines()[2:]
lib = MyExtension('libmseed',
                  define_macros=[('WIN32', sys.platform == 'win32')],
                  libraries=[],
                  sources=[src + 'fileutils.c', src + 'genutils.c',
                           src + 'gswap.c', src + 'lmplatform.c',
                           src + 'lookup.c', src + 'msrutils.c',
                           src + 'pack.c', src + 'packdata.c',
                           src + 'traceutils.c', src + 'tracelist.c',
                           src + 'unpack.c', src + 'unpackdata.c',
                           src + 'logging.c'],
                  export_symbols=symbols,
                  extra_link_args=[])


setup(
    name='obspy.mseed',
    version=VERSION,
    description="Read & write seismograms, Format MiniSeed",
    long_description="""
    obspy.mseed - Read & write seismograms, Format MiniSeed
    
    This module contains Python wrappers for libmseed - The MiniSeed
    library of Chad Trabant. Libmseed is written in C and interfaced via
    Python ctypes.

    For more information visit http://www.obspy.org.
    """,
    url='http://www.obspy.org',
    author='The ObsPy Development Team & Chad Trabant',
    author_email='devs@obspy.org',
    classifiers=[],
    keywords=['ObsPy', 'seismology', 'MSEED', 'MiniSEED', 'waveform',
              'seismograms'],
    license='GPL2',
    packages=find_packages(),
    namespace_packages=['obspy'],
    zip_safe=True,
    requires=[
        'setuptools',
        'obspy.core(>=0.2)',
        'numpy',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de" + \
        "/svn/obspy/obspy.mseed/trunk#egg=obspy.mseed-dev",
    platforms=['any'],
    ext_package='obspy.mseed.lib',
    ext_modules=[lib],
    include_package_data=True,
    test_suite="obspy.mseed.tests.suite",
)
