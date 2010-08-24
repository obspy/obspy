#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.signal installer

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from setuptools import find_packages, setup
from setuptools.extension import Extension
import os, numpy as np
import platform


VERSION = open(os.path.join("obspy", "signal", "VERSION.txt")).read()


# hack to prevent build_ext from trying to append "init" to the export symbols
class finallist(list):
    def append(self, object):
        return

class MyExtension(Extension):
    def __init__(self, *args, **kwargs):
        Extension.__init__(self, *args, **kwargs)
        self.export_symbols = finallist(self.export_symbols)

macros = []
#if platform.system() == "Windows":
#    # disable some warnings for MSVC
#    macros.append(('_CRT_SECURE_NO_WARNINGS', '1'))

src = os.path.join('obspy', 'signal', 'src') + os.sep
symbols = [s.strip() for s in open(src + 'libsignal.def', 'r').readlines()[2:]
           if s.strip() != '']

# try to find platfrom independently the suffix of fftpack_lite
numpy_lib_dir = os.path.dirname(np.fft.__file__)
libraries = []
extra_link_args = []
for ext in ('dylib', 'so', 'dll', 'pyd'):
    fftpack = 'fftpack_lite.%s' % ext
    if not os.path.exists(os.path.join(numpy_lib_dir, fftpack)):
        continue
    if platform.system() == 'Darwin':
        extra_link_args=['-dylib_file libfftpack_lite.dylib:%s/%s' % \
                         (numpy_lib_dir, fftpack)]
    else:
        libraries.append(':'+fftpack)
    break

if platform.system() == "Windows":
    lib = MyExtension('libsignal',
                      define_macros=macros,
                      #library_dirs=[numpy_lib_dir],
                      #libraries=libraries,
                      sources=[src + 'recstalta.c', src + 'xcorr.c',
                               src + 'coordtrans.c', src + 'pk_mbaer.c',
                               src + 'filt_util.c', src + 'arpicker.c'],
                               #src + 'bbfk.c', src + 'runtimelink.c'],
                      export_symbols=symbols,
                      extra_link_args=extra_link_args)
else:
    lib = MyExtension('libsignal',
                      define_macros=macros,
                      library_dirs=[numpy_lib_dir],
                      libraries=libraries,
                      sources=[src + 'recstalta.c', src + 'xcorr.c',
                               src + 'coordtrans.c', src + 'pk_mbaer.c',
                               src + 'filt_util.c', src + 'arpicker.c',
                               src + 'bbfk.c'],
                      export_symbols=symbols,
                      extra_link_args=extra_link_args)


# setup
setup(
    name='obspy.signal',
    version=VERSION,
    description="Python signal processing routines for seismology.",
    long_description="""
    obspy.signal - Python signal processing routines for seismology

    Capabilities include filtering, triggering, rotation, instrument
    correction and coordinate transformations.

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
    keywords=['ObsPy', 'seismology', 'signal', 'processing', 'filter',
              'trigger', 'instrument correction', 'picker',
              'instrument simulation', 'features', 'envelope', 'hob'],
    packages=find_packages(),
    namespace_packages=['obspy'],
    zip_safe=False,
    install_requires=[
        'setuptools',
        'obspy.core',
        'scipy',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de" + \
        "/svn/obspy/obspy.signal/trunk#egg=obspy.signal-dev",
    ext_package='obspy.signal.lib',
    ext_modules=[lib],
    test_suite="obspy.signal.tests.suite",
    include_package_data=True,
)
