#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
GSE2 and GSE1 read and write support for ObsPy.

This module provides read and write support for GSE2 CM6 compressed as well as
GSE1 ASCII waveform data and header info. Most methods are based on the C
library GSE_UTI of Stefan Stange, which is interfaced via Python ctypes.

ObsPy is an open-source project dedicated to provide a Python framework for
processing seismological data. It provides parsers for common file formats and
seismological signal processing routines which allow the manipulation of
seismological time series (see Beyreuther et al. 2010, Megies et al. 2011).
The goal of the ObsPy project is to facilitate rapid application development
for seismology.

For more information visit http://www.obspy.org.

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Stefan Stange
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from setuptools import find_packages, setup
from setuptools.extension import Extension
import os
import platform
import shutil
import sys

LOCAL_PATH = os.path.abspath(os.path.dirname(__file__))
DOCSTRING = __doc__.split("\n")

# package specific
VERSION = open(os.path.join(LOCAL_PATH, 'obspy', 'gse2', 'VERSION.txt')).read()
NAME = 'obspy.gse2'
AUTHOR = 'The ObsPy Development Team & Stefan Stange'
AUTHOR_EMAIL = 'devs@obspy.org'
LICENSE = 'GNU Lesser General Public License, Version 3 (LGPLv3)'
KEYWORDS = ['ObsPy', 'seismology', 'GSE2', 'waveform', 'seismograms']
INSTALL_REQUIRES = ['obspy.core']
ENTRY_POINTS = {
    'obspy.plugin.waveform': [
        'GSE2 = obspy.gse2.core',
        'GSE1 = obspy.gse2.core',
    ],
    'obspy.plugin.waveform.GSE2': [
        'isFormat = obspy.gse2.core:isGSE2',
        'readFormat = obspy.gse2.core:readGSE2',
        'writeFormat = obspy.gse2.core:writeGSE2',
    ],
    'obspy.plugin.waveform.GSE1': [
        'isFormat = obspy.gse2.core:isGSE1',
        'readFormat = obspy.gse2.core:readGSE1',
    ],
}


def setupLibGSE2():
    """
    Prepare building of C extension libgse2.
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
    src = os.path.join('obspy', 'gse2', 'src', 'GSE_UTI') + os.sep
    symbols = [s.strip()
               for s in open(src + 'gse_functions.def').readlines()[2:]
               if s.strip() != '']
    # system specific settings
    if platform.system() == "Windows":
        # disable some warnings for MSVC
        macros.append(('_CRT_SECURE_NO_WARNINGS', '1'))
    # create library name
    if 'develop' in sys.argv:
        lib_name = 'libgse2-%s-%s-py%s' % (
            platform.system(), platform.architecture()[0],
            ''.join([str(i) for i in platform.python_version_tuple()[:2]]))
    else:
        lib_name = 'libgse2'
    # setup C extension
    lib = MyExtension(lib_name,
                      define_macros=macros,
                      libraries=[],
                      sources=[src + 'buf.c', src + 'gse_functions.c'],
                      export_symbols=symbols,
                      extra_link_args=[])
    return lib


def convert2to3():
    """
    Convert source to Python 3.x syntax using lib2to3.
    """
    # create a new 2to3 directory for converted source files
    dst_path = os.path.join(LOCAL_PATH, '2to3')
    shutil.rmtree(dst_path, ignore_errors=True)

    # copy original tree into 2to3 folder ignoring some unneeded files
    def ignored_files(_adir, filenames):
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
        download_url="https://svn.obspy.org/trunk/%s#egg=%s-dev" % (NAME,
                                                                    NAME),
        include_package_data=True,
        test_suite="%s.tests.suite" % (NAME),
        entry_points=ENTRY_POINTS,
        ext_package='obspy.gse2.lib',
        ext_modules=[setupLibGSE2()],
        use_2to3=True,
    )
    # cleanup after using lib2to3 for Python 3.x
    if sys.version_info[0] == 3:
        os.chdir(LOCAL_PATH)


if __name__ == '__main__':
    setupPackage()
