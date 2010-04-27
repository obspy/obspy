#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.xseed installer

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from setuptools import find_packages, setup
import os


VERSION = open(os.path.join("obspy", "xseed", "VERSION.txt")).read()


setup(
    name='obspy.xseed',
    version=VERSION,
    description="Tool to convert between Dataless SEED and XML-SEED files.",
    long_description="""
    obspy.xseed - Tool to convert between Dataless SEED and XML-SEED files.

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
    keywords=['ObsPy', 'seismology', 'SEED', 'Dataless SEED', 'XML-SEED',
              'XSEED'],
    packages=find_packages(),
    namespace_packages=['obspy'],
    zip_safe=False,
    install_requires=[
        'setuptools',
        'lxml',
        'obspy.core',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de" + \
        "/svn/obspy/obspy.xseed/trunk#egg=obspy.xseed-dev",
    include_package_data=True,
    test_suite="obspy.xseed.tests.suite",
    entry_points={
        'console_scripts': [
            'obspy-dataless2xseed = obspy.xseed.scripts.dataless2xseed:main',
            'obspy-xseed2dataless = obspy.xseed.scripts.xseed2dataless:main',
            'obspy-dataless2resp = obspy.xseed.scripts.dataless2resp:main',
        ],
    },
)
