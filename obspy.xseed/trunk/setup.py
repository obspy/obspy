#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.xseed installer

@copyright: The ObsPy Development Team (devs@obspy.org)
@license: GNU Lesser General Public License, Version 3 (LGPLv3)
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
    classifiers=[],
    keywords=['ObsPy', 'seismology', 'SEED', 'Dataless SEED', 'XML-SEED',
              'XSEED'],
    license='LGPLv3',
    packages=find_packages(),
    namespace_packages=['obspy'],
    zip_safe=True,
    install_requires=[
        'setuptools',
        'lxml<2.2.3',
        'obspy.core>0.2.1',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de" + \
        "/svn/obspy/obspy.xseed/trunk#egg=obspy.xseed-dev",
    platforms=['any'],
    include_package_data=True,
    test_suite="obspy.xseed.tests.suite",
    entry_points={
        'console_scripts': [
            'dataless2xseed = obspy.xseed.scripts.dataless2xseed:main',
            'xseed2dataless = obspy.xseed.scripts.xseed2dataless:main',
        ],
    },
)
