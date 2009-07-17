# -*- coding: utf-8 -*-
"""
setup.py bdist_egg
"""

from setuptools import setup, find_packages

version = '0.1.0'

GPL2 = """
GNU General Public License (GPL)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
02110-1301, USA.
"""

setup(
    name='obspy.gse2',
    version=version,
    description="Read & Write Seismograms, Format GSE2.",
    long_description="""
    obspy.gse2 - Read & Write Seismograms, Format GSE2.

    This module contains python wrappers for gse_functions - The GSE2 library
    of Stefan Stange. Currently CM6 compressed GSE2 files are supported, this
    should be sufficient for most cases. Gse_functions are written in C and
    interfaced via python-ctypes.
    See: http://www.orfeus-eu.org/Software/softwarelib.html#gse
    """,
    classifiers=[],
    keywords='Seismology, GSE2',
    author='Moritz Beyreuther, Stefan Stange',
    author_email='beyreuth@geophysik.uni-muenchen.de',
    url='https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.gse2',
    license=GPL2,
    packages=find_packages(exclude=['ez_setup']),
    namespace_packages=['obspy'],
    include_package_data=True,
    zip_safe=False,
    test_suite="obspy.gse2.tests.suite",
    install_requires=[
        'obspy.core',
        'setuptools',
        'numpy',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.gse2/trunk#egg=obspy.gse2-dev",
    dependency_links=[
        "https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.core/trunk#egg=obspy.core-dev"
    ],
)
