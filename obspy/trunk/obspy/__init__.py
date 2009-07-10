# -*- coding: utf-8 -*-
"""
ObsPy, Python for Seismological Observatories

Capabilities include filtering, instrument simulation, unified read and
write support for GSE2, MSEED, SAC, triggering, imaging, xseed and
there is experimental support to arclink and seishub.

Note currently only obspy.core, obspy.gse2, obspy.mseed and
obspy.imaging are automatically installed. Other modules can be found
at www.obspy.org


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

# A namespace package will not have any method, class or any other
# objects defined in that level.
# @see: http://baijum81.livejournal.com/22412.html

# DON'T IMPORT HERE ANYTHING!

# See http://peak.telecommunity.com/DevCenter/setuptools#namespace-packages
try:
    __import__('pkg_resources').declare_namespace(__name__)
except ImportError:
    from pkgutil import extend_path
    __path__ = extend_path(__path__, __name__)
