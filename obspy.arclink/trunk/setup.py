#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.arclink installer

@copyright: The ObsPy Development Team (devs@obspy.org)
@license: GNU Lesser General Public License, Version 3 (LGPLv3)
"""

from setuptools import find_packages, setup


VERSION = '0.2.2'


setup(
    name='obspy.arclink',
    version=VERSION,
    description="ArcLink/WebDC request client for of ObsPy.",
    long_description="""
    obspy.arclink - ArcLink/WebDC request client for of ObsPy.

    For more information visit http://www.obspy.org.
    """,
    url='http://www.obspy.org',
    author='The ObsPy Development Team',
    author_email='devs@obspy.org',
    classifiers=[],
    keywords=['ObsPy', 'Seismology', 'ArcLink', 'MiniSEED', 'SEED',
              'Inventory', 'Waveform'],
    license='LGPLv3',
    packages=find_packages(),
    namespace_packages=['obspy'],
    zip_safe=True,
    install_requires=[
        'setuptools',
        'obspy.mseed>0.2.1',
        'lxml',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de" + \
        "/svn/obspy/obspy.arclink/trunk#egg=obspy.arclink-dev",
    platforms=['any'],
    include_package_data=True,
    test_suite="obspy.arclink.tests.suite",
)
