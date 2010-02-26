#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.gse2 installer

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Stefan Stange
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
from setuptools.extension import Extension
import os


VERSION = open(os.path.join("obspy", "gse2", "VERSION.txt")).read()


# hack to prevent build_ext from trying to append "init" to the export symbols
class finallist(list):
    def append(self, object):
        return

class MyExtension(Extension):
    def __init__(self, *args, **kwargs):
        Extension.__init__(self, *args, **kwargs)
        self.export_symbols = finallist(self.export_symbols)

src = os.path.join('obspy', 'gse2', 'src', 'GSE_UTI') + os.sep
symbols = [s.strip()
           for s in open(src + 'gse_functions.def', 'r').readlines()[2:]
           if s.strip() != '']
lib = MyExtension('gse_functions',
                  define_macros=[],
                  libraries=[],
                  sources=[src + 'buf.c', src + 'gse_functions.c'],
                  export_symbols=symbols,
                  extra_link_args=[])


setup(
    name='obspy.gse2',
    version=VERSION,
    description="Read & write seismograms, Format GSE2.",
    long_description="""
    obspy.gse2 - Read & write seismograms, Format GSE2.

    This module contains Python wrappers for gse_functions - The GSE2 library
    of Stefan Stange (http://www.orfeus-eu.org/Software/softwarelib.html#gse).
    Currently CM6 compressed GSE2 files are supported, this should be 
    sufficient for most cases. Gse_functions are written in C and interfaced 
    via Python ctypes.

    For more information visit http://www.obspy.org.
    """,
    url='http://www.obspy.org',
    author='The ObsPy Development Team & Stefan Stange',
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
    keywords=['ObsPy', 'seismology', 'GSE2', 'waveform', 'seismograms'],
    packages=find_packages(),
    namespace_packages=['obspy'],
    zip_safe=True,
    install_requires=[
        'setuptools',
        'obspy.core',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de" + \
        "/svn/obspy/obspy.gse2/trunk#egg=obspy.gse2-dev",
    ext_package='obspy.gse2.lib',
    ext_modules=[lib],
    include_package_data=True,
    test_suite="obspy.gse2.tests.suite",
    entry_points="""
        [obspy.plugin.waveform]
        GSE2 = obspy.gse2.core

        [obspy.plugin.waveform.GSE2]
        isFormat = obspy.gse2.core:isGSE2
        readFormat = obspy.gse2.core:readGSE2
        writeFormat = obspy.gse2.core:writeGSE2
    """,
)
