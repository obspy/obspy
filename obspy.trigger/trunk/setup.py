# -*- coding: utf-8 -*-
"""
setup.py bdist_egg
"""

from setuptools import setup, find_packages

version = '0.0.1'

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
    name='obspy.trigger',
    version=version,
    description="Python trigger routines for seismology.",
    long_description="""
    obspy.trigger - Python trigger routines for seismology.
    """,
    classifiers=[],
    keywords='Seismology, Triggers',
    author='Moritz Beyreuther',
    author_email='beyreuth@geophysik.uni-muenchen.de',
    url='https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.trigger',
    license=GPL2,
    packages=find_packages(exclude=['ez_setup']),
    namespace_packages=['obspy'],
    include_package_data=True,
    zip_safe=False,
    # test_suite = "obspy.picker.tests.suite",
    install_requires=[
        'setuptools',
        # -*- Extra requirements: -*
    ],
    download_url="https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.trigger/trunk#egg=obspy.trigger-dev",
    dependency_links = [
    ],
)
