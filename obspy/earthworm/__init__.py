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
(1) :meth:`~obspy.earthworm.client.Client.getWaveform()`: The following example
    illustrates how to request and plot 30 seconds of all three broadband
    channels (``"BH*"``) of station ``"TUCA"`` of the `Pacific Northwest
    Seismic Network <http://www.pnsn.org/>`_ (``"UW"``).

    >>> from obspy.earthworm import Client
    >>> client = Client("pele.ess.washington.edu", 16017)
    >>> response = client.availability("UW", "TUCA", channel="BHZ")
    >>> print response  # doctest: #SKIP
    [('UW',
      'TUCA',
      '--',
      'BHZ',
      UTCDateTime(2011, 11, 27, 0, 0, 0, 525000),
      UTCDateTime(2011, 12, 29, 20, 50, 31, 525000))]
    >>> t = response[0][4]
    >>> st = client.getWaveform('UW', 'TUCA', '', 'BH*', t + 100, t + 130)
    >>> st.plot()  # doctest: +SKIP

    .. plot::

        from obspy.earthworm import Client
        from obspy import UTCDateTime
        client = Client("pele.ess.washington.edu", 16017)
        response = client.availability("UW", "TUCA", channel="BHZ")
        t = response[0][4]
        st = client.getWaveform('UW', 'TUCA', '', 'BH*', t + 100, t + 130)
        st.plot()
"""

from client import Client


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
