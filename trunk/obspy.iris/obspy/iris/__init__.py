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
(1) In the following example the :meth:`~obspy.iris.client.Client.getWaveform()`
    method is used to request and plot 60 minutes of the BHZ channel of station
    Albuquerque, New Mexico ("ANMO") of the Global Seismograph Network ("IU")
    for an seismic event around 2010-02-27 06:30 (UTC).

    >>> from obspy.iris import Client
    >>> from obspy.core import UTCDateTime
    >>> client = Client()
    >>> t = UTCDateTime("2010-02-27T06:45:00.000")
    >>> st = client.getWaveform("IU", "ANMO", "00", "BHZ", t, t + 60 * 60)
    >>> st.plot() #doctest: +SKIP

    .. plot::

        from obspy.core import UTCDateTime
        from obspy.iris import Client
        client = Client()
        t = UTCDateTime("2010-02-27T06:30:00.000")
        st = client.getWaveform("IU", "ANMO", "00", "BHZ", t, t + 60 * 60)
        st.plot() #doctest: +SKIP

(2) Alternatively you may save any requested waveform unmodified into your local
    file system using the :meth:`~obspy.iris.client.Client.saveWaveform()`
    method.

    >>> from obspy.iris import Client
    >>> from obspy.core import UTCDateTime
    >>> client = Client()
    >>> t1 = UTCDateTime("2010-02-27T06:30:00.000")
    >>> t2 = UTCDateTime("2010-02-27T07:30:00.000")
    >>> client.saveWaveform("IU", "ANMO", "00", "BHZ", t1, t2) #doctest: +SKIP

(3) The :meth:`~obspy.iris.client.Client.saveResponse()` method writes the
    response information into a file. You may choose one of the format
    RESP or StationXML.

    >>> from obspy.iris import Client
    >>> from obspy.core import UTCDateTime
    >>> client = Client()
    >>> t = UTCDateTime(2009, 1, 1)
    >>> client.saveResponse('resp.txt', 'IU', 'ANMO', '', '*',
    ...                     t, t + 1, format="RESP") #doctest: +SKIP

Each of the following methods directly wrap a single Web service provided by
IRIS (http://www.iris.edu/ws/):
  * :meth:`~obspy.iris.client.Client.resp()` - provides access to channel
    response information in the SEED RESP format (as used by evalresp)
  * :meth:`~obspy.iris.client.Client.station()` - provides access to station
    metadata in the IRIS DMC database
  * :meth:`~obspy.iris.client.Client.dataselect()` - returns a single channel
    of time series data
  * :meth:`~obspy.iris.client.Client.bulkdataselect()` - returns multiple
    channels of time series data for specified time ranges
  * :meth:`~obspy.iris.client.Client.availability()` - returns information about
    what time series data is available at the IRIS-DMC
  * :meth:`~obspy.iris.client.Client.sacpz()` - provides access to instrument
    response information (per-channel) as poles and zeros in the ASCII format
    used by SAC and other programs

Please see the documentation for each method for further information and
examples to retrieve various data from the IRIS DMC.
"""

from obspy.core.util import _getVersionString
from client import Client


__version__ = _getVersionString("obspy.iris")


if __name__ == '__main__': # pragma: no cover
    import doctest
    doctest.testmod(exclude_empty=True)
