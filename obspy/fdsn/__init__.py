# -*- coding: utf-8 -*-
"""
obspy.fdsn - FDSN Web service client for ObsPy
==============================================
The obspy.fdsn package contains a client to access web servers that implement
the FDSN web service definitions (http://www.fdsn.org/webservices/).

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

Basic Usage
-----------

All examples make use of the FDSN Web Service at IRIS. Other FDSN Web Service
providers are available too, see :meth:`~obspy.fdsn.client.Client.__init__()`.

(1) :meth:`~obspy.fdsn.client.Client.get_waveform()`: The following example
    illustrates how to request and plot 60 minutes of the ``"BHZ"`` channel of
    station Albuquerque, New Mexico (``"ANMO"``) of the Global Seismograph
    Network (``"IU"``) for an seismic event around 2010-02-27 06:45 (UTC).
    Results are returned as a :class:`~obspy.core.stream.Stream` object.
    For how to send multiple requests simultaneously (avoiding network
    overhead) see :meth:`~obspy.fdsn.client.Client.get_waveform_bulk()`

    >>> from obspy.fdsn import Client
    >>> from obspy import UTCDateTime
    >>> client = Client()
    >>> t = UTCDateTime("2010-02-27T06:45:00.000")
    >>> st = client.get_waveform("IU", "ANMO", "00", "BHZ", t, t + 60 * 60)
    >>> st.plot()  # doctest: +SKIP

    .. plot::

        from obspy import UTCDateTime
        from obspy.fdsn import Client
        client = Client()
        t = UTCDateTime("2010-02-27T06:45:00.000")
        st = client.get_waveform("IU", "ANMO", "00", "BHZ", t, t + 60 * 60)
        st.plot()

(2) :meth:`~obspy.fdsn.client.Client.get_events()`: Retrieves event data from
    the server. Results are returned as a :class:`~obspy.core.event.Catalog`)
    object.

    >>> client = Client()
    >>> starttime = UTCDateTime("2011-04-01")
    >>> endtime = UTCDateTime("2011-04-15")
    >>> cat = client.get_events(starttime=starttime, endtime=endtime,
    ...                         minmagnitude=6.7)
    >>> print(cat)  # doctest: +NORMALIZE_WHITESPACE
    2 Event(s) in Catalog:
    2011-04-07T14:32:43.290000Z | +38.276, +141.588 | 7.1 MW
    2011-04-03T20:06:40.390000Z |  -9.848, +107.693 | 6.7 MW
    >>> cat.plot()  # doctest: +SKIP

    .. plot::

        from obspy import UTCDateTime
        from obspy.fdsn import Client
        client = Client()
        starttime = UTCDateTime("2011-04-01")
        endtime = UTCDateTime("2011-04-15")
        cat = client.get_events(starttime=starttime, endtime=endtime, \
                                minmag=6.7)
        cat.plot()

(3) :meth:`~obspy.fdsn.client.Client.get_stations()`: Retrieves station data
    from the server. Results are returned as a StationXML string (will be
    changed to an Obspy Inventory object in the near future).

    >>> client = Client()
    >>> stationxml_string = client.get_stations(
    ...     latitude=-56.1, longitude=-26.7, maxradius=15)
    >>> for line in stationxml_string.splitlines()[:6]:
    ...     print line  # doctest: +ELLIPSIS
    <?xml version="1.0" encoding="ISO-8859-1"?>
    <BLANKLINE>
    <FDSNStationXML ...>
     <Source>IRIS-DMC</Source>
     <Sender>IRIS-DMC</Sender>
     <Module>IRIS WEB SERVICE: fdsnws-station | version: 1.0.7</Module>

Please see the documentation for each method for further information and
examples.
"""

from client import Client


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
