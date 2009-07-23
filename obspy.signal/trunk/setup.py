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
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
USA.
"""
setup(
    name='obspy.signal',
    version=version,
    description="Python signal processing routines for seismology.",
    long_description="""
    obspy.signal - Python signal processing routines for seismology.

    Capabilities include filtering, triggering, rotation, instrument
    correction and coordinate transformations.
    """,
    classifiers=[],
    keywords='Seismology Signal, Filter, Triggers, Instrument Correction',
    author='Tobias Megies, Moritz Beyreuther, Yannik Behr',
    author_email='megies@geophysik.uni-muenchen.de',
    url='https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.signal',
    license=GPL2,
    packages=find_packages(exclude=['ez_setup']),
    namespace_packages=['obspy'],
    include_package_data=True,
    zip_safe=True,
    test_suite="obspy.signal.tests.suite",
    install_requires=[
        'obspy.core',
        'setuptools',
        'scipy',
        # -*- Extra requirements: -*
    ],
    download_url="https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.signal/trunk#egg=obspy.signal-dev",
    dependency_links=[
        "https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.core/trunk#egg=obspy.core"
    ],
)
