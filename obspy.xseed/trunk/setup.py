#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.xseed installer

:copyright: The ObsPy Development Team (devs@obspy.org)
:license: GNU Lesser General Public License, Version 3 (LGPLv3)
"""

from setuptools import find_packages, setup


VERSION = '0.2.2'


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
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: ' + \
        'GNU Library or Lesser General Public License (LGPL)',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Geophysics',
    ],
    keywords=['ObsPy', 'seismology', 'SEED', 'Dataless SEED', 'XML-SEED',
              'XSEED'],
    packages=find_packages(),
    namespace_packages=['obspy'],
    zip_safe=True,
    install_requires=[
        'setuptools',
        'lxml',
        'obspy.core>0.2.1',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de" + \
        "/svn/obspy/obspy.xseed/trunk#egg=obspy.xseed-dev",
    include_package_data=True,
    test_suite="obspy.xseed.tests.suite",
    entry_points={
        'console_scripts': [
            'dataless2xseed = obspy.xseed.scripts.dataless2xseed:main',
            'xseed2dataless = obspy.xseed.scripts.xseed2dataless:main',
        ],
    },
)
