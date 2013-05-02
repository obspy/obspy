# -*- coding: utf-8 -*-
"""
obspy.neic CWB Query module for ObsPy.
==============================================
The obspy.neic package contains a client for the NEIC CWB Query server.  A publci
server is at 137.227.224.97 (cwbpub.cr.usgs.gov) on port 2061.

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


Please see the documentation for each method for further information and
examples to retrieve various data from the IRIS DMC.
"""

from client import Client


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
