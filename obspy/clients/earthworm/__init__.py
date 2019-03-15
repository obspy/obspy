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
(1) :meth:`~obspy.clients.earthworm.client.Client.getWaveform()`: The following
    example illustrates how to request and plot 30 seconds of all three
    short period channels (``"EH*"``) of station ``"ACH"`` of the `Alaska
    Volcano Observatory <https://www.avo.alaska.edu/>`_ (``"AV"``).

    >>> from obspy.clients.earthworm import Client
    >>> client = Client("pubavo1.wr.usgs.gov", 16022)
    >>> response = client.get_availability('AV', 'ACH', channel='BHE')
    >>> print(response)  # doctest: +SKIP
    [('AV',
      'ACH',
      '--',
      'BHE',
      UTCDateTime(2015, 1, 22, 7, 26, 32, 679000),
      UTCDateTime(2015, 3, 23, 7, 26, 29, 919966)]
    >>> t = response[0][4]
    >>> st = client.get_waveforms('AV', 'ACH', '', 'BH*', t + 100, t + 130)
    >>> st.plot()  # doctest: +SKIP

    .. plot::

        from obspy.clients.earthworm import Client
        from obspy import UTCDateTime
        client = Client("pubavo1.wr.usgs.gov", 16022, timeout=5)
        response = client.get_availability('AV', 'ACH', channel='EHE')
        t = response[0][4]
        st = client.get_waveforms('AV', 'ACH', '', 'BH*', t + 100, t + 130)
        st.plot()
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy.clients.earthworm.client import Client  # NOQA


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
