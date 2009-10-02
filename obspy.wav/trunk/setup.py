# -*- coding: utf-8 -*-
"""
setup.py bdist_egg
"""

from setuptools import setup, find_packages

version = '0.1.2'

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
    name='obspy.wav',
    version=version,
    description="Read & Write Seismograms to audio, Format WAV.",
    long_description="""
    obspy.wav - Read & Write Seismograms to audio, Format WAV
    
    Python method in order to read and write seismograms to WAV audio
    files. The data are squeezed to audible frequencies.

    For more information visit http://www.obspy.org.
    """,
    classifiers=[],
    keywords='ObsPy, Seismology, WAV',
    author='The ObsPy Development Team',
    author_email='beyreuth@geophysik.uni-muenchen.de',
    url='https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.wav',
    license='GPL',
    packages=find_packages(exclude=['ez_setup']),
    namespace_packages=['obspy'],
    include_package_data=True,
    zip_safe=False,
    test_suite="obspy.wav.tests.suite",
    install_requires=[
        'obspy.core',
        'setuptools',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.wav/trunk#egg=obspy.wav-dev",
    dependency_links=[
        "https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.core/trunk#egg=obspy.core-dev"
    ],
)
