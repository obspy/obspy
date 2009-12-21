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
    entry_points="""
        [obspy.plugin.waveform]
        GSE2 = obspy.gse2.core

        [obspy.plugin.waveform.GSE2]
        isFormat = obspy.gse2.core:isGSE2
        readFormat = obspy.gse2.core:readGSE2
        writeFormat = obspy.gse2.core:writeGSE2
    """,
)
