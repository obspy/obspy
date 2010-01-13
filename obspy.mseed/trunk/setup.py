#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.mseed installer

:copyright: The ObsPy Development Team (devs@obspy.org)
:license: GNU General Public License (GPL)
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
import platform


VERSION = '0.2.2'


# hack to prevent build_ext from trying to append "init" to the export symbols
class finallist(list):
    def append(self, object):
        return

class MyExtension(Extension):
    def __init__(self, *args, **kwargs):
        Extension.__init__(self, *args, **kwargs)
        self.export_symbols = finallist(self.export_symbols)

macros = []
if platform.system() == "Windows":
    macros.append(('WIN32', '1'))
    # The following lines are needed for Python 2.6 compiled with MinGW 
    # otherwise exception is raised by C extension using gmtime or localtime
    # :see: http://bugs.python.org/issue3308
    macros.append(('time_t', '__int64'))
    macros.append(('localtime', '_localtime64'))
    macros.append(('gmtime', '_gmtime64'))

src = os.path.join('obspy', 'mseed', 'src', 'libmseed') + os.sep
symbols = open(src + 'libmseed.def', 'r').readlines()[2:]
lib = MyExtension('libmseed',
                  define_macros=macros,
                  libraries=[],
                  sources=[src + 'fileutils.c', src + 'genutils.c',
                           src + 'gswap.c', src + 'lmplatform.c',
                           src + 'lookup.c', src + 'msrutils.c',
                           src + 'pack.c', src + 'packdata.c',
                           src + 'traceutils.c', src + 'tracelist.c',
                           src + 'unpack.c', src + 'unpackdata.c',
                           src + 'selection.c', src + 'logging.c'],
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
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Programming Language :: C',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Geophysics',
    ],
    keywords=['ObsPy', 'seismology', 'MSEED', 'MiniSEED', 'waveform',
              'seismograms'],
    packages=find_packages(),
    namespace_packages=['obspy'],
    zip_safe=True,
    install_requires=[
        'setuptools',
        'obspy.core>0.2.1',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de" + \
        "/svn/obspy/obspy.mseed/trunk#egg=obspy.mseed-dev",
    ext_package='obspy.mseed.lib',
    ext_modules=[lib],
    include_package_data=True,
    test_suite="obspy.mseed.tests.suite",
    entry_points="""
        [obspy.plugin.waveform]
        MSEED = obspy.mseed.core

        [obspy.plugin.waveform.MSEED]
        isFormat = obspy.mseed.core:isMSEED
        readFormat = obspy.mseed.core:readMSEED
        writeFormat = obspy.mseed.core:writeMSEED
    """,
)
