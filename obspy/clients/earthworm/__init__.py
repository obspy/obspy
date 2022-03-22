# -*- coding: utf-8 -*-
"""
obspy.clients.earthworm - Earthworm Wave Server client for ObsPy.
=================================================================

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Victor Kress
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

Basic Usage
-----------
(1) :meth:`~obspy.clients.earthworm.client.Client.get_waveforms()`: The
    following example illustrates how to request and plot 30 seconds of all
    three short period channels (``"EH*"``) of station ``"KCG"`` of the `Alaska
    Volcano Observatory <https://www.avo.alaska.edu/>`_ (``"AV"``).

    >>> from obspy.clients.earthworm import Client
    >>> client = Client("pubavo1.wr.usgs.gov", 16022)
    >>> response = client.get_availability('AV', 'AKV', channel='BHE')
    >>> print(response)  # doctest: +SKIP
    [('AV',
      'AKV',
      '--',
      'BHE',
      UTCDateTime(2021, 10, 30, 12, 2, 27, 473000),
      UTCDateTime(2021, 12, 29, 12, 2, 16, 899000)]
    >>> t = response[0][4]
    >>> st = client.get_waveforms('AV', 'AKV', '', 'BH*', t + 100, t + 130)
    >>> st.plot()  # doctest: +SKIP

    .. plot::

        from obspy.clients.earthworm import Client
        from obspy import UTCDateTime
        client = Client("pubavo1.wr.usgs.gov", 16022, timeout=5)
        response = client.get_availability('AV', 'AKV', channel='BHE')
        t = response[0][4]
        st = client.get_waveforms('AV', 'AKV', '', 'BH*', t + 100, t + 130)
        st.plot()
"""
from obspy.clients.earthworm.client import Client  # NOQA


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
