"""
setup.py bdist_egg
"""

from setuptools import setup, find_packages

version = '0.1.3'

GPL2 = """
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
    name='obspy',
    version=version,
    description="ObsPy, Python for Seismological Observatories",
    long_description="""
    ObsPy, Python for Seismological Observatories

    Capabilities include filtering, instrument simulation, unified read and
    write support for GSE2, MSEED, SAC, triggering, imaging, XML-SEED and
    there is experimental support for Arclink and SeisHub servers.

    Note currently only obspy.core, obspy.gse2, obspy.mseed and
    obspy.imaging are automatically installed.

    For more information visit http://www.obspy.org.
    """,
    classifiers=[],
    keywords='ObsPy, Seismology',
    author='The ObsPy Development Team',
    author_email='beyreuth@geophysik.uni-muenchen.de',
    url='https://svn.geophysik.uni-muenchen.de/svn/obspy',
    license=GPL2,
    packages=find_packages(exclude=['ez_setup']),
    namespace_packages=[],
    include_package_data=True,
    zip_safe=True,
    test_suite="obspy.core.testing.suite",
    install_requires=[
        'setuptools',
        'numpy',
        'matplotlib',
        'obspy.core',
        'obspy.gse2',
        'obspy.mseed',
        'obspy.sac',
        'obspy.imaging'
    ],
    download_url="https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy/trunk#egg=obspy-dev",
    dependency_links=[
        "https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.core/trunk#egg=obspy.core-dev",
        "https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.gse2/trunk#egg=obspy.gse2-dev",
        "https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.mseed/trunk#egg=obspy.mseed-dev",
        "https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.imaging/trunk#egg=obspy.imaging-dev",
    ],
)
