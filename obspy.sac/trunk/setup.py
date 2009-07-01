# -*- coding: utf-8 -*-
#
# Copyright (c) 2008 by:
#     * Yannik Behr
#
# GNU General Public License (GPL)
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
# 02110-1301, USA.

"""
setup.py bdist_egg
"""

from setuptools import setup, find_packages

version = '0.0.1'


setup(
    name='obspy.sac',
    version=version,
    description="Read and write support for SAC files (seismograms).",
    long_description="""""",
    classifiers=[],
    keywords='Seismology SAC',
    author='Yannik Behr',
    author_email='yannik.behr@vuw.ac.nz',
    url='https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.sac',
    license='GPL',
    packages=find_packages(exclude=['ez_setup']),
    namespace_packages=['obspy'],
    include_package_data=True,
    zip_safe=True,
    # test_suite = "obspy.sac.tests",
    install_requires=[
        'obspy.core',
        'setuptools',
        # -*- Extra requirements: -*
    ],
    dependency_links = [
        "https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.core/trunk#egg=obspy.core"
    ],
)
