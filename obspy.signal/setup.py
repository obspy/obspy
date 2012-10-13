#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Signal processing routines for ObsPy.

The obspy.signal package contains signal processing routines for seismology.
Capabilities include filtering, triggering, rotation, instrument correction and
coordinate transformations.

ObsPy is an open-source project dedicated to provide a Python framework for
processing seismological data. It provides parsers for common file formats and
seismological signal processing routines which allow the manipulation of
seismological time series (see Beyreuther et al. 2010, Megies et al. 2011).
The goal of the ObsPy project is to facilitate rapid application development
for seismology.

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
import glob

LOCAL_PATH = os.path.abspath(os.path.dirname(__file__))
DOCSTRING = __doc__.split("\n")

# package specific
VERSION = open(os.path.join(LOCAL_PATH, 'obspy', 'signal',
                            'VERSION.txt')).read()
NAME = 'obspy.signal'
AUTHOR = 'The ObsPy Development Team'
AUTHOR_EMAIL = 'devs@obspy.org'
LICENSE = 'GNU Lesser General Public License, Version 3 (LGPLv3)'
KEYWORDS = ['ObsPy', 'seismology', 'signal', 'processing', 'filter', 'trigger',
            'instrument correction', 'picker', 'instrument simulation',
            'features', 'envelope', 'hob']
INSTALL_REQUIRES = ['obspy.core > 0.6.1', 'scipy']
ENTRY_POINTS = {
    'obspy.plugin.trigger': [
        'recstalta = obspy.signal.trigger:recSTALTA',
        'carlstatrig = obspy.signal.trigger:carlSTATrig',
        'classicstalta = obspy.signal.trigger:classicSTALTA',
        'delayedstalta = obspy.signal.trigger:delayedSTALTA',
        'zdetect = obspy.signal.trigger:zDetect',
        'recstaltapy = obspy.signal.trigger:recSTALTAPy',
        'classicstaltapy = obspy.signal.trigger:classicSTALTAPy',
    ],
    'obspy.plugin.filter': [
        'bandpass = obspy.signal.filter:bandpass',
        'bandstop = obspy.signal.filter:bandstop',
        'lowpass = obspy.signal.filter:lowpass',
        'highpass = obspy.signal.filter:highpass',
        'lowpassCheby2 = obspy.signal.filter:lowpassCheby2',
        'lowpassFIR = obspy.signal.filter:lowpassFIR',
        'remezFIR = obspy.signal.filter:remezFIR',
    ],
    'obspy.plugin.detrend': [
        'simple = obspy.signal.detrend:simple',
    ],
    'obspy.plugin.taper': [
        'cosine = obspy.signal.invsim:cosTaper',
    ],
}


def setupLibSignal():
    """
    Prepare building of C extension libsignal.
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
            ''.join([str(i) for i in platform.python_version_tuple()[:2]]))
    else:
        lib_name = 'libsignal'
    # setup C extension
    lib = MyExtension(lib_name,
                      define_macros=macros,
                      include_dirs=[numpy_include_dir],
                      sources=[src + 'recstalta.c', src + 'xcorr.c',
                               src + 'coordtrans.c', src + 'pk_mbaer.c',
                               src + 'filt_util.c', src + 'arpicker.c',
                               src + 'bbfk.c', src + 'stalta.c',
                               src_fft + 'fftpack.c',
                               src_fft + 'fftpack_litemodule.c'],
                      export_symbols=symbols)
    return lib


def setupLibEvalResp():
    """
    Prepare building of evalresp extension library.
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
    src = os.path.join('obspy', 'signal', 'src') + os.sep
    src_evresp = os.path.join('obspy', 'signal', 'src', 'evalresp') + os.sep
    evresp_include_dir = src_evresp
    symbols = [s.strip() for s in open(src + 'libevresp.def').readlines()[2:]
               if s.strip() != '']
    # system specific settings
    if platform.system() == "Windows":
        # needed by evalresp evresp.h
        macros.append(('WIN32', '1'))
        # disable some warnings for MSVC
        macros.append(('_CRT_SECURE_NO_WARNINGS', '1'))
    # create library name
    if 'develop' in sys.argv:
        lib_name = 'libevresp-%s-%s-py%s' % (
            platform.system(), platform.architecture()[0],
            ''.join([str(i) for i in platform.python_version_tuple()[:2]]))
    else:
        lib_name = 'libevresp'
    # setup C extension
    lib = MyExtension(lib_name,
                      define_macros=macros,
                      include_dirs=[evresp_include_dir],
                      sources=glob.glob(os.path.join(src_evresp, '*.c')),
                      export_symbols=symbols)
    return lib


def convert2to3():
    """
    Convert source to Python 3.x syntax using lib2to3.
    """
    # create a new 2to3 directory for converted source files
    dst_path = os.path.join(LOCAL_PATH, '2to3')
    shutil.rmtree(dst_path, ignore_errors=True)
    # copy original tree into 2to3 folder ignoring some unneeded files

    def ignored_files(adir, filenames):  # @UnusedVariable
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
        ext_package='obspy.signal.lib',
        ext_modules=[setupLibSignal(), setupLibEvalResp()],
        use_2to3=True,
    )
    # cleanup after using lib2to3 for Python 3.x
    if sys.version_info[0] == 3:
        os.chdir(LOCAL_PATH)


if __name__ == '__main__':
    setupPackage()
