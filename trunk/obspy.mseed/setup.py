#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mini-SEED read and write support for ObsPy.

This module provides read and write support for Mini-SEED waveform data and
some other convenient methods to handle Mini-SEED files. Most methods are based
on libmseed, a C library framework by Chad Trabant and interfaced via python
ctypes.

ObsPy is an open-source project dedicated to provide a Python framework for
processing seismological data. It provides parsers for common file formats and
seismological signal processing routines which allow the manipulation of
seismological time series (see Beyreuther et al. 2010, Megies et al. 2011).
The goal of the ObsPy project is to facilitate rapid application development
for seismology.

For more information visit http://www.obspy.org.

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Chad Trabant
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

from distutils.ccompiler import get_default_compiler
from setuptools import find_packages, setup
from setuptools.extension import Extension
import os
import platform
import shutil
import sys

LOCAL_PATH = os.path.abspath(os.path.dirname(__file__))
DOCSTRING = __doc__.split("\n")

# package specific
VERSION = open(os.path.join(LOCAL_PATH, 'obspy', 'mseed',
                            'VERSION.txt')).read()
NAME = 'obspy.mseed'
AUTHOR = 'The ObsPy Development Team & Chad Trabant'
AUTHOR_EMAIL = 'devs@obspy.org'
LICENSE = 'GNU General Public License (GPL)'
KEYWORDS = ['ObsPy', 'seismology', 'MSEED', 'MiniSEED', 'waveform',
            'seismograms']
INSTALL_REQUIRES = ['obspy.core']
ENTRY_POINTS = {
    'obspy.plugin.waveform': [
        'MSEED = obspy.mseed.core',
    ],
    'obspy.plugin.waveform.MSEED': [
        'isFormat = obspy.mseed.core:isMSEED',
        'readFormat = obspy.mseed.core:readMSEED',
        'writeFormat = obspy.mseed.core:writeMSEED',
    ],
}


def setupLibMSEED():
    """
    Prepare building of C extension libmseed.
    """
    # hack to prevent build_ext to append __init__ to the export symbols
    class finallist(list):
        def append(self, object):
            return

    class MyExtension(Extension):
        def __init__(self, *args, **kwargs):
            Extension.__init__(self, *args, **kwargs)
            self.export_symbols = finallist(self.export_symbols)
    macros = []
    extra_link_args = []
    extra_compile_args = []
    src = os.path.join('obspy', 'mseed', 'src', 'libmseed') + os.sep
    symbols = [s.strip()
               for s in open(src + 'libmseed.def', 'r').readlines()[2:]
               if s.strip() != '']

    # system specific settings
    if platform.system() == "Windows":
        # needed by libmseed lmplatform.h
        macros.append(('WIN32', '1'))
        # disable some warnings for MSVC
        macros.append(('_CRT_SECURE_NO_WARNINGS', '1'))
        if 'msvc' in sys.argv or \
            ('-c' not in sys.argv and get_default_compiler() == 'msvc'):
            if platform.architecture()[0] == '32bit':
                # Workaround Win32 and MSVC - see issue #64
                extra_compile_args.append("/fp:strict")

    # create library name
    if 'develop' in sys.argv:
        lib_name = 'libmseed-%s-%s-py%s' % (
            platform.system(), platform.architecture()[0],
            ''.join([str(i) for i in platform.python_version_tuple()[:2]]))
    else:
        lib_name = 'libmseed'

    # setup C extension
    lib = MyExtension(lib_name,
                      define_macros=macros,
                      libraries=[],
                      sources=[src + 'fileutils.c', src + 'genutils.c',
                               src + 'gswap.c', src + 'lmplatform.c',
                               src + 'lookup.c', src + 'msrutils.c',
                               src + 'pack.c', src + 'packdata.c',
                               src + 'traceutils.c', src + 'tracelist.c',
                               src + 'unpack.c', src + 'unpackdata.c',
                               src + 'selection.c', src + 'logging.c',
                               src + 'obspy-readbuffer.c',
                               src + 'parseutils.c'],
                      export_symbols=symbols,
                      extra_link_args=extra_link_args,
                      extra_compile_args=extra_compile_args)
    return lib


def convert2to3():
    """
    Convert source to Python 3.x syntax using lib2to3.
    """
    # create a new 2to3 directory for converted source files
    dst_path = os.path.join(LOCAL_PATH, '2to3')
    shutil.rmtree(dst_path, ignore_errors=True)
    # copy original tree into 2to3 folder ignoring some unneeded files
    def ignored_files(adir, filenames):
        return ['.svn', '2to3', 'debian', 'build', 'dist'] + \
               [fn for fn in filenames if fn.startswith('distribute')] + \
               [fn for fn in filenames if fn.endswith('.egg-info')]
    shutil.copytree(LOCAL_PATH, dst_path, ignore=ignored_files)
    os.chdir(dst_path)
    sys.path.insert(0, dst_path)
    # run lib2to3 script on duplicated source
    from lib2to3.main import main
    print("Converting to Python3 via lib2to3...")
    main("lib2to3.fixes", ["-w", "-n", "--no-diffs", "obspy"])


def getVersion():
    # fetch version
    file = os.path.join(LOCAL_PATH, 'obspy', NAME.split('.')[1], 'VERSION.txt')
    return open(file).read()


def setupPackage():
    # use lib2to3 for Python 3.x
    if sys.version_info[0] == 3:
        convert2to3()
    # setup package
    setup(
        name=NAME,
        version=getVersion(),
        description=DOCSTRING[1],
        long_description="\n".join(DOCSTRING[3:]),
        url="http://www.obspy.org",
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        license=LICENSE,
        platforms='OS Independent',
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: GNU Library or ' + \
                'Lesser General Public License (LGPL)',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Physics'],
        keywords=KEYWORDS,
        packages=find_packages(exclude=['distribute_setup']),
        namespace_packages=['obspy'],
        zip_safe=False,
        install_requires=INSTALL_REQUIRES,
        download_url="https://svn.obspy.org/trunk/%s#egg=%s-dev" % (NAME, NAME),
        include_package_data=True,
        test_suite="%s.tests.suite" % (NAME),
        entry_points=ENTRY_POINTS,
        ext_package='obspy.mseed.lib',
        ext_modules=[setupLibMSEED()],
        use_2to3=True,
    )
    # cleanup after using lib2to3 for Python 3.x
    if sys.version_info[0] == 3:
        os.chdir(LOCAL_PATH)


if __name__ == '__main__':
    setupPackage()