# -*- coding: utf-8 -*-
#
# Copyright (c) 2008 by:
#     * Moritz Beyreuther
#     * Stefan Stange
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
    name='obspy.gse2',
    version=version,
    description="Read and write support for GSE2 files (seismograms).",
    long_description="""""",
    classifiers=[],
    keywords='Seismology GSE2',
    author='Moritz Beyreuther, Stefan Stange',
    author_email='beyreuth@geophysik.uni-muenchen.de',
    url='https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.gse2',
    license='GPL',
    packages=find_packages(exclude=['ez_setup']),
    namespace_packages=['obspy'],
    include_package_data=True,
    zip_safe=False,
    # test_suite = "obspy.gse2.tests",
    install_requires=[
        'obspy',
        'setuptools',
        #'numpy',
        # -*- Extra requirements: -*
    ],
    dependency_links = [
        "https://svn.geophysik.uni-muenchen.de/svn/obspy/trunk"
    ],
)
