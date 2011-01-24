#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Python signal processing routines for seismology.

The obspy.signal package contains signal processing routines for seismology.
Capabilities include filtering, triggering, rotation, instrument correction and
coordinate transformations.

ObsPy is an open-source project dedicated to provide a Python framework for
processing seismological data. It provides parsers for common file formats and
seismological signal processing routines which allow the manipulation of
seismological time series (see  Beyreuther et. al. 2010). The goal of the ObsPy
project is to facilitate rapid application development for seismology. 

For more information visit http://www.obspy.org.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from setuptools import find_packages, setup
from setuptools.extension import Extension
import numpy as np
import os
import platform
import shutil
import sys


# release specific
LOCAL_PATH = os.path.abspath(os.path.dirname(__file__))
VERSION = open(os.path.join(LOCAL_PATH, 'obspy', 'signal', 'VERSION.txt')).read()

# package specific
NAME = 'obspy.signal'
AUTHOR = 'The ObsPy Development Team'
AUTHOR_EMAIL = 'devs@obspy.org'
LICENSE = 'GNU Lesser General Public License, Version 3 (LGPLv3)'
KEYWORDS = ['ObsPy', 'seismology', 'signal', 'processing', 'filter', 'trigger',
            'instrument correction', 'picker', 'instrument simulation',
            'features', 'envelope', 'hob']
INSTALL_REQUIRES = ['setuptools', 'obspy.core', 'scipy']
ENTRY_POINTS = {}

# package independent
DOCLINES = __doc__.split("\n")
DESCRIPTION = DOCLINES[0]
LONG_DESCRIPTION = "\n".join(DOCLINES[2:])
URL = "http://www.obspy.org"
DOWNLOAD_URL = "https://svn.obspy.org/trunk/%s#egg=%s-dev" % (NAME, NAME)
PLATFORMS = 'OS Independent'
ZIP_SAFE = False
CLASSIFIERS = filter(None, """
Development Status :: 4 - Beta
Environment :: Console
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)
Operating System :: OS Independent
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Physics
""".split('\n'))


def setupLibSignal():
    # hack to prevent build_ext to append __init__ to the export symbols
    class finallist(list):
        def append(self, object):
            return

    class MyExtension(Extension):
        def __init__(self, *args, **kwargs):
            Extension.__init__(self, *args, **kwargs)
            self.export_symbols = finallist(self.export_symbols)
    macros = []
    src = os.path.join('obspy', 'signal', 'src') + os.sep
    src_fft = os.path.join('obspy', 'signal', 'src', 'fft') + os.sep
    numpy_include_dir = os.path.join(os.path.dirname(np.core.__file__),
                                     'include')
    symbols = [s.strip() for s in open(src + 'libsignal.def').readlines()[2:]
               if s.strip() != '']
    # system specific settings
    if platform.system() == "Windows":
        # disable some warnings for MSVC
        macros.append(('_CRT_SECURE_NO_WARNINGS', '1'))
    # create library name
    if 'develop' in sys.argv:
        lib_name = 'libsignal-%s-%s-py%s' % (
            platform.system(), platform.architecture()[0],
            ''.join(platform.python_version_tuple()[:2]))
    else:
        lib_name = 'libsignal'
    # setup C extension
    lib = MyExtension(lib_name,
                      define_macros=macros,
                      include_dirs=[numpy_include_dir],
                      sources=[src + 'recstalta.c', src + 'xcorr.c',
                               src + 'coordtrans.c', src + 'pk_mbaer.c',
                               src + 'filt_util.c', src + 'arpicker.c',
                               src + 'bbfk.c', src_fft + 'fftpack.c',
                               src_fft + 'fftpack_litemodule.c'],
                      export_symbols=symbols)
    return lib


def setupPackage():
    # Perform 2to3 if needed
    if sys.version_info[0] == 3:
        dst_path = os.path.join(LOCAL_PATH, '2to3')
        shutil.rmtree(dst_path, ignore_errors=True)
        def ignored_files(adir, filenames):
            return ['.svn', '2to3', 'debian', 'docs'] + \
                   [fn for fn in filenames if fn.startswith('distribute')] + \
                   [fn for fn in filenames if fn.endswith('.egg-info')]
        shutil.copytree(LOCAL_PATH, dst_path, ignore=ignored_files)
        os.chdir(dst_path)
        sys.path.insert(0, dst_path)
        from lib2to3.main import main
        print("Converting to Python3 via lib2to3...")
        main("lib2to3.fixes", ["-w", "-n", "--no-diffs", "obspy"])
    try:
        setup(
            name=NAME,
            version=VERSION,
            description=DESCRIPTION,
            long_description=LONG_DESCRIPTION,
            url=URL,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            license=LICENSE,
            platforms=PLATFORMS,
            classifiers=CLASSIFIERS,
            keywords=KEYWORDS,
            packages=find_packages(exclude=['distribute_setup']),
            namespace_packages=['obspy'],
            zip_safe=ZIP_SAFE,
            install_requires=INSTALL_REQUIRES,
            download_url=DOWNLOAD_URL,
            include_package_data=True,
            test_suite="%s.tests.suite" % (NAME),
            entry_points=ENTRY_POINTS,
            ext_package='obspy.signal.lib',
            ext_modules=[setupLibSignal()],
        )
    finally:
        if sys.version_info[0] == 3:
            del sys.path[0]
            os.chdir(LOCAL_PATH)
    return


if __name__ == '__main__':
    setupPackage()
