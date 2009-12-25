# -*- coding: utf-8 -*-
"""
setup.py bdist_egg
"""

from setuptools import setup, find_packages

VERSION = '0.2.2'

GPL2 = """
GNU General Public License (GPL)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
USA.
"""

setup(
    name='obspy.imaging',
    version=VERSION,
    description="Provides tools for displaying features used in seismology.",
    long_description="""
    obspy.imaging - Provides tools for displaying features used in seismology.

    For more information visit http://www.obspy.org.
    """,
    classifiers=[],
    keywords='ObsPy, Seismology, Imaging, Beachball, Focal Mechanism, Waveform, Spectogram',
    author='The ObsPy Development Team',
    author_email='barsch@geophysik.uni-muenchen.de',
    url='https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.imaging',
    license=GPL2,
    packages=find_packages(exclude=['ez_setup']),
    namespace_packages=['obspy'],
    include_package_data=True,
    zip_safe=False,
    test_suite="obspy.imaging.tests.suite",
    install_requires=[
        'setuptools',
        'obspy.core>0.2.1'
        'matplotlib',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.imaging/trunk#egg=obspy.imaging-dev",
    dependency_links=[
        "https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.core/trunk#egg=obspy.core-dev"
    ],
)
