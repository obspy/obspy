#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.gse2 installer

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


VERSION = open(os.path.join("obspy", "gse2", "VERSION.txt")).read()


# hack to prevent build_ext from trying to append "init" to the export symbols
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
           for s in open(src + 'gse_functions.def', 'r').readlines()[2:]
           if s.strip() != '']

# system specific settings
if platform.system() == "Windows":
    # disable some warnings for MSVC
    macros.append(('_CRT_SECURE_NO_WARNINGS', '1'))

# create library name
python_version = '_'.join(platform.python_version_tuple())
lib_name = 'libgse2-%s-%s-%s-py%s' % (platform.node(),
                                      platform.platform(terse=1),
                                      platform.architecture()[0],
                                      python_version)

# setup C extension
lib = MyExtension(lib_name,
                  define_macros=macros,
                  libraries=[],
                  sources=[src + 'buf.c', src + 'gse_functions.c'],
                  export_symbols=symbols,
                  extra_link_args=[])


setup(
    name='obspy.gse2',
    version=VERSION,
    description="Read & write seismograms, Format GSE2.",
    long_description="""
    obspy.gse2 - Read & write seismograms, Format GSE2

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
    zip_safe=False,
    install_requires=[
        'setuptools',
        'obspy.core',
    ],
    download_url="https://svn.obspy.org/obspy.gse2/trunk#egg=obspy.gse2-dev",
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
