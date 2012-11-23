# -*- coding: utf-8 -*-
"""
obspy.iris - IRIS Web service client for ObsPy
==============================================
The obspy.iris package contains a client for the DMC Web services provided by
IRIS (http://www.iris.edu/ws/).

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

Basic Usage
-----------
(1) :meth:`~obspy.iris.client.Client.getWaveform()`: The following example
    illustrates how to request and plot 60 minutes of the ``"BHZ"`` channel of
    station Albuquerque, New Mexico (``"ANMO"``) of the Global Seismograph
    Network (``"IU"``) for an seismic event around 2010-02-27 06:45 (UTC).

    >>> from obspy.iris import Client
    >>> from obspy import UTCDateTime
    >>> client = Client()
    >>> t = UTCDateTime("2010-02-27T06:45:00.000")
    >>> st = client.getWaveform("IU", "ANMO", "00", "BHZ", t, t + 60 * 60)
    >>> st.plot()  # doctest: +SKIP

    .. plot::

        from obspy import UTCDateTime
        from obspy.iris import Client
        client = Client()
        t = UTCDateTime("2010-02-27T06:45:00.000")
        st = client.getWaveform("IU", "ANMO", "00", "BHZ", t, t + 60 * 60)
        st.plot()

(2) :meth:`~obspy.iris.client.Client.saveWaveform()`: Writes the requested
    waveform unmodified into your local file system. Here we request a Full
    SEED volume.

    >>> from obspy.iris import Client
    >>> from obspy import UTCDateTime
    >>> client = Client()
    >>> t1 = UTCDateTime("2010-02-27T06:30:00.000")
    >>> t2 = UTCDateTime("2010-02-27T07:30:00.000")
    >>> client.saveWaveform('IU.ANMO.00.BHZ.mseed', 'IU', 'ANMO',
    ...                     '00', 'BHZ', t1, t2) # doctest: +SKIP

(3) :meth:`~obspy.iris.client.Client.saveResponse()`: Writes the response
    information into a file. You may choose one of the format
    ``'RESP'``, ``'StationXML'`` or ``'SACPZ'``.

    >>> from obspy.iris import Client
    >>> from obspy import UTCDateTime
    >>> client = Client()
    >>> t = UTCDateTime(2009, 1, 1)
    >>> client.saveResponse('resp.txt', 'IU', 'ANMO', '', '*',
    ...                     t, t + 1, format="RESP") #doctest: +SKIP

(4) :meth:`~obspy.iris.client.Client.getEvents()`: Retrieves event data from
    IRIS. You may choose one of the format ``'xml'`` (QuakeML format) or
    ``'catalog'`` (:class:`~obspy.core.event.Catalog`).

    >>> client = Client()
    >>> starttime = UTCDateTime("2011-04-01")
    >>> endtime = UTCDateTime("2011-04-15")
    >>> cat = client.getEvents(starttime=starttime, endtime=endtime,
    ...                        minmag=6.7)
    >>> print(cat)  # doctest: +NORMALIZE_WHITESPACE
    4 Event(s) in Catalog:
    2011-04-11T08:16:12.730000Z | +37.001, +140.401 | 6.7 MW
    2011-04-07T14:32:43.290000Z | +38.276, +141.588 | 7.1 MW
    2011-04-07T13:11:22.400000Z | +17.208,  -94.338 | 6.7 MW
    2011-04-03T20:06:40.390000Z |  -9.848, +107.693 | 6.7 MW
    >>> cat.plot()  # doctest: +SKIP

    .. plot::

        from obspy import UTCDateTime
        from obspy.iris import Client
        client = Client()
        starttime = UTCDateTime("2011-04-01")
        endtime = UTCDateTime("2011-04-15")
        cat = client.getEvents(starttime=starttime, endtime=endtime, \
                               minmag=6.7)
        cat.plot()


Low-level Web service Interfaces
--------------------------------

Each of the following methods directly wrap a single Web service provided by
IRIS (http://www.iris.edu/ws/):

**Request Tools**

* :meth:`~obspy.iris.client.Client.availability()` - returns information about
  what time series data is available at the IRIS DMC
* :meth:`~obspy.iris.client.Client.bulkdataselect()` - returns multiple
  channels of time series data for specified time ranges
* :meth:`~obspy.iris.client.Client.dataselect()` - returns a single channel
  of time series data
* :meth:`~obspy.iris.client.Client.evalresp()` - evaluates instrument response
  information stored at the IRIS DMC and outputs ASCII data or
  `Bode Plots <http://en.wikipedia.org/wiki/Bode_plots>`_.
* :meth:`~obspy.iris.client.Client.event()` - returns event information in the
  `QuakeML <https://quake.ethz.ch/quakeml/>`_ format. Events may be selected
  based on location, time, catalog, contributor and internal identifiers.
* :meth:`~obspy.iris.client.Client.resp()` - provides access to channel
  response information in the SEED RESP format (as used by evalresp)
* :meth:`~obspy.iris.client.Client.sacpz()` - provides access to instrument
  response information (per-channel) as poles and zeros in the ASCII format
  used by SAC and other programs
* :meth:`~obspy.iris.client.Client.station()` - provides access to station
  metadata in the IRIS DMC database
* :meth:`~obspy.iris.client.Client.timeseries()` - fetches segments of seismic
  data and returns data formatted in either MiniSEED, ASCII or SAC. It can
  optionally filter the data.

**Calculation Tools**

* :meth:`~obspy.iris.client.Client.traveltime()` - calculates travel-times for
  seismic phases using a 1-D spherical earth model.
* :meth:`~obspy.iris.client.Client.distaz()` - calculate the distance and
  azimuth between two points on a sphere.
* :meth:`~obspy.iris.client.Client.flinnengdahl()` - converts a latitude,
  longitude pair into either a Flinn-Engdahl seismic region code or region
  name.


Please see the documentation for each method for further information and
examples to retrieve various data from the IRIS DMC.
"""

from client import Client


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
