# -*- coding: utf-8 -*-
"""
obspy.earthworm - Earthworm Wave Server client for ObsPy.
=========================================================

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Victor Kress
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

Basic Usage
-----------
(1) :meth:`~obspy.earthworm.client.Client.getWaveform()`: The following example
    illustrates how to request and plot 30 seconds of all three short period
    channels (``"EH*"``) of station ``"ACH"`` of the `Alaska Volcano
    Observatory <https://www.avo.alaska.edu/>`_ (``"AV"``).

    >>> from obspy.earthworm import Client
    >>> client = Client("pubavo1.wr.usgs.gov", 16022)
    >>> response = client.availability('AV', 'ACH', channel='EHE')
    >>> print(response)  # doctest: +SKIP
    [('AV',
      'ACH',
      '--',
      'EHE',
      UTCDateTime(2015, 1, 22, 7, 26, 32, 679000),
      UTCDateTime(2015, 3, 23, 7, 26, 29, 919966)]
    >>> t = response[0][4]
    >>> st = client.getWaveform('AV', 'ACH', '--', 'EH*', t + 100, t + 130)
    >>> st.plot()  # doctest: +SKIP

    .. plot::

        from obspy.earthworm import Client
        from obspy import UTCDateTime
        client = Client("pubavo1.wr.usgs.gov", 16022, timeout=5)
        response = client.availability('AV', 'ACH', channel='EHE')
        t = response[0][4]
        st = client.getWaveform('AV', 'ACH', '--', 'EH*', t + 100, t + 130)
        st.plot()
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from .client import Client  # NOQA


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
