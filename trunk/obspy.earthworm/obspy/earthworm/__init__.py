# -*- coding: utf-8 -*-
"""
obspy.earthworm - Earthworm Wave Server client for ObsPy.
=========================================================

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Victor Kress
:license:
    GNU General Public License (GPLv2)
    (http://www.gnu.org/licenses/old-licenses/gpl-2.0.html)

Basic Usage
-----------
The example illustrates how to request and plot 30 seconds of all three
short period channels (``"BH*"``) of station ``"LON"`` of the Pacific
Northwest Seismic Network (http://www.pnsn.org/).

        .. rubric:: Example

        >>> from obspy.earthworm import Client
        >>> from obspy.core import UTCDateTime
        >>> client = Client("hood.ess.washington.edu", 16021)
        >>> dt = UTCDateTime() - 2000  # now - 2000 seconds
        >>> st = client.getWaveform('UW', 'LON', '', 'BH*', dt, dt + 30)
        >>> st.plot()  # doctest: +SKIP

        .. plot::

            from obspy.earthworm import Client
            from obspy.core import UTCDateTime
            client = Client("hood.ess.washington.edu", 16021)
            dt = UTCDateTime() - 2000
            st = client.getWaveform('UW', 'LON', '', 'BH*', dt, dt + 30)
            st.plot()
"""

from obspy.core.util import _getVersionString
from client import Client


__version__ = _getVersionString("obspy.earthworm")


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
