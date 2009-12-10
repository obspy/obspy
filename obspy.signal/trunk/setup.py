#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.wav installer

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


VERSION = '0.2.0'


# hack to prevent build_ext from trying to append "init" to the export symbols
class finallist(list):
    def append(self, object):
        return

class MyExtension(Extension):
    def __init__(self, *args, **kwargs):
        Extension.__init__(self, *args, **kwargs)
        self.export_symbols = finallist(self.export_symbols)

src = os.path.join('obspy', 'signal', 'src') + os.sep
symbols = open(src + 'signal.def', 'r').readlines()[2:]
lib = MyExtension('signal',
                  define_macros=[],
                  libraries=[],
                  sources=[src + 'recstalta.c', src + 'xcorr.c',
                           src + 'coordtrans.c', src + 'pk_mbaer.c',
                           src + 'filt_util.c', src + 'arpicker.c'],
                  export_symbols=symbols,
                  extra_link_args=[])


# setup
setup(
    name='obspy.signal',
    version=VERSION,
    description="Python signal processing routines for seismology.",
    long_description="""
    obspy.signal - Python signal processing routines for seismology.

    Capabilities include filtering, triggering, rotation, instrument
    correction and coordinate transformations.

    For more information visit http://www.obspy.org.
    """,
    url='http://www.obspy.org',
    author='The ObsPy Development Team',
    author_email='devs@obspy.org',
    classifiers=[],
    keywords=['ObsPy', 'seismology', 'signal', 'filter', 'triggers',
              'instrument correction', ],
    license='GPL2',
    packages=find_packages(),
    namespace_packages=['obspy'],
    zip_safe=True,
    requires=[
        'setuptools',
        'obspy.core(>=0.2)',
        'scipy',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de" + \
        "/svn/obspy/obspy.signal/trunk#egg=obspy.signal-dev",
    platforms=['any'],
    ext_package='obspy.signal.lib',
    ext_modules=[lib],
    test_suite="obspy.signal.tests.suite",
    include_package_data=True,
)
