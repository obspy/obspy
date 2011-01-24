#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""ObsPy core classes, Python for Seismological Observatories

The obspy.core package contains common methods and classes for ObsPy. It
includes UTCDateTime, Stats, Stream and Trace classes and methods for reading 
seismograms.

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
import distribute_setup
import os
import sys
import shutil


# release specific
LOCAL_PATH = os.path.abspath(os.path.dirname(__file__))
VERSION = open(os.path.join(LOCAL_PATH, 'obspy', 'core', 'VERSION.txt')).read()

# package specific
NAME = 'obspy.core'
AUTHOR = 'The ObsPy Development Team'
AUTHOR_EMAIL = 'devs@obspy.org'
LICENSE = 'GNU Lesser General Public License, Version 3 (LGPLv3)'
KEYWORDS = ['ObsPy', 'seismology']
INSTALL_REQUIRES = ['numpy>1.0.0']
ENTRY_POINTS = {
    'console_scripts': [
        'obspy-runtests = obspy.core.scripts.runtests:main',
    ],
    'obspy.plugin.waveform': [
        'TSPAIR = obspy.core.ascii',
        'SLIST = obspy.core.ascii',
    ],
    'obspy.plugin.waveform.TSPAIR': [
        'isFormat = obspy.core.ascii:isTSPAIR',
        'readFormat = obspy.core.ascii:readTSPAIR',
    ],
    'obspy.plugin.waveform.SLIST': [
        'isFormat = obspy.core.ascii:isSLIST',
        'readFormat = obspy.core.ascii:readSLIST',
    ],
}

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


def setupPackage():
    #
    distribute_setup.use_setuptools()
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
        )
    finally:
        if sys.version_info[0] == 3:
            del sys.path[0]
            os.chdir(LOCAL_PATH)
    return


if __name__ == '__main__':
    setupPackage()
