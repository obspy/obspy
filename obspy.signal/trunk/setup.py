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
import os


VERSION = open(os.path.join("obspy", "signal", "VERSION.txt")).read()



# hack to prevent build_ext from trying to append "init" to the export symbols
class finallist(list):
    def append(self, object):
        return

class MyExtension(Extension):
    def __init__(self, *args, **kwargs):
        Extension.__init__(self, *args, **kwargs)
        self.export_symbols = finallist(self.export_symbols)

src = os.path.join('obspy', 'signal', 'src') + os.sep
symbols = [s.strip() for s in open(src + 'libsignal.def', 'r').readlines()[2:]]
lib = MyExtension('libsignal',
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
        'Topic :: Scientific/Engineering :: Geophysics',
    ],
    keywords=['ObsPy', 'seismology', 'signal', 'filter', 'triggers',
              'instrument correction', ],
    packages=find_packages(),
    namespace_packages=['obspy'],
    zip_safe=True,
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
