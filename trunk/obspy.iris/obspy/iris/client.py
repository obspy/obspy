# -*- coding: utf-8 -*-
"""
IRIS web service client for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from obspy.core import UTCDateTime, read, Stream
from obspy.core.util import NamedTemporaryFile, BAND_CODE
from urllib2 import HTTPError
import os
import sys
import urllib
import urllib2


class Client(object):
    """
    IRIS web service request client.

    Examples
    --------

    >>> from obspy.iris import Client
    >>> from obspy.core import UTCDateTime
    >>> client = Client()
       
    >>> t = UTCDateTime("2010-02-27T06:30:00.000")
    >>> st = client.getWaveform("IU", "ANMO", "00", "BHZ", t, t + 20)
    >>> print(st)
    1 Trace(s) in Stream:
    IU.ANMO.00.BHZ | 2010-02-27T06:30:00.019538Z - 2010-02-27T06:30:20.019538Z | 20.0 Hz, 401 samples
    """
    def __init__(self, base_url="http://www.iris.edu/ws",
                 user="", password="", timeout=10):
        self.base_url = base_url
        self.timeout = timeout
        # Create an OpenerDirector for Basic HTTP Authentication
        password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, base_url, user, password)
        auth_handler = urllib2.HTTPBasicAuthHandler(password_mgr)
        opener = urllib2.build_opener(auth_handler)
        # install globally
        urllib2.install_opener(opener)

    def _fetch(self, url, **params):
        # replace special characters 
        remoteaddr = self.base_url + url + '?' + urllib.urlencode(params)
        # timeout exists only for Python >= 2.6
        if sys.hexversion < 0x02060000:
            response = urllib2.urlopen(remoteaddr)
        else:
            response = urllib2.urlopen(remoteaddr, timeout=self.timeout)
        doc = response.read()
        return doc

    def _HTTP_request(self, url, data, headers={}):
        """
        Send a HTTP request via urllib2.

        :type url: String
        :param url: Complete URL of resource
        :type data: String
        :param data: Channel list as returned by `availability`-webservice
        :type headers: dict
        :param headers: Additional header information for request
        """
        headers['User-Agent'] = "ObsPy"
        # replace special characters 
        remoteaddr = self.base_url + url
        req = urllib2.Request(url=remoteaddr, data=data, headers=headers)
        # timeout exists only for Python >= 2.6
        if sys.hexversion < 0x02060000:
            response = urllib2.urlopen(req)
        else:
            response = urllib2.urlopen(req, timeout=self.timeout)
        data = response.read()
        return data

    def getWaveform(self, network, station, location, channel, starttime,
                     endtime, quality='B'):
        """
        Gets a ObsPy Stream object.
        Wildcards are allowed for `network`, `station`, `location` and
        `channel`.

        Example
        -------

        >>> from obspy.iris import Client
        >>> from obspy.core import UTCDateTime
        >>> client = Client()
           
        >>> t1 = UTCDateTime("2010-02-27T06:30:00.000")
        >>> t2 = UTCDateTime("2010-02-27T10:30:00.000")
        >>> st = client.getWaveform("IU", "ANMO", "00", "BHZ", t1, t2)
        >>> print st
        1 Trace(s) in Stream:
        IU.ANMO.00.BHZ | 2010-02-27T06:30:00.019538Z - 2010-02-27T10:30:00.019538Z | 20.0 Hz, 288001 samples

        >>> t1 = UTCDateTime("2010-084T00:00:00")
        >>> t2 = UTCDateTime("2010-084T00:30:00")
        >>> st = client.getWaveform("TA", "A25A", "", "BH*", t1, t2)
        >>> print st
        3 Trace(s) in Stream:
        TA.A25A..BHE | 2010-03-25T00:00:00.000000Z - 2010-03-25T00:30:00.000000Z | 40.0 Hz, 72001 samples
        TA.A25A..BHN | 2010-03-25T00:00:00.000000Z - 2010-03-25T00:30:00.000000Z | 40.0 Hz, 72001 samples
        TA.A25A..BHZ | 2010-03-25T00:00:00.000000Z - 2010-03-25T00:30:00.000000Z | 40.0 Hz, 72001 samples


        Parameters
        ----------
        network : string
            Network code, e.g. 'IU' or 'I*'.
        station : string
            Station code, e.g. 'ANMO' or 'A*'.
        location : string
            Location code, e.g. '00' or '*'.
        channel : string
            Channel code, e.g. 'BHZ' or 'B*'.
        starttime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            Start date and time.
        endtime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            End date and time.
        quality : 'D', 'R', 'Q', 'M' or 'B', optional
            MiniSEED data quality indicator. M and B (default) are treated the
            same and indicate best available. If M or B are selected, the
            output data records will be stamped with a M.

        Returns
        -------
            :class:`~obspy.core.stream.Stream`
        """
        kwargs = {}
        kwargs['network'] = str(network)[0:2]
        kwargs['station'] = str(station)[0:5]
        if location:
            kwargs['location'] = str(location)[0:2]
        else:
            kwargs['location'] = '--'
        kwargs['channel'] = str(channel)[0:3]
        # try to be intelligent in starttime/endtime extension for fetching data
        try:
            t_extension = 2.0 / BAND_CODE[kwargs['channel'][0]]
        except:
            # use 1 second extension if no proper bandcode info
            t_extension = 1.0
        kwargs['starttime'] = UTCDateTime(starttime) - t_extension
        kwargs['endtime'] = UTCDateTime(endtime) + t_extension
        if str(quality).upper() in ['D', 'R', 'Q', 'M', 'B']:
            kwargs['quality'] = str(quality).upper()

        # single channel request, go via `dataselect`-webservice
        if all([val.isalnum() for val in (kwargs['network'], kwargs['station'], kwargs['location'], kwargs['channel'])]):
            st = self.dataselect(**kwargs)
        # wildcarded channel request, go via `availability`+`bulkdataselect`-webservices
        else:
            quality = kwargs.pop("quality", "")
            bulk = self.availability(**kwargs)
            st = self.bulkdataselect(bulk, quality)

        st.trim(UTCDateTime(starttime), UTCDateTime(endtime))
        return st


    def dataselect(self, **kwargs):
        """
        Interface for `dataselect`-webservice of IRIS
        (http://www.iris.edu/ws/dataselect/).
        Single channel request, no wildcards allowed.
        This webservice can be used via
        :meth:`~obspy.iris.client.Client.getWaveform`.

        Example
        -------

        >>> from obspy.iris import Client
        >>> from obspy.core import UTCDateTime
        >>> client = Client()
           
        >>> t1 = UTCDateTime("2010-02-27T06:30:00.000")
        >>> t2 = UTCDateTime("2010-02-27T10:30:00.000")
        >>> st = client.dataselect(network="IU", station="ANMO", location="00",
        ...         channel="BHZ", starttime=t1, endtime=t2)
        >>> print st
        1 Trace(s) in Stream:
        IU.ANMO.00.BHZ | 2010-02-27T06:30:00.019538Z - 2010-02-27T10:29:59.969538Z | 20.0 Hz, 288000 samples

        Parameters
        ----------
        network : string
            Network code, e.g. 'BW'.
        station : string
            Station code, e.g. 'MANZ'.
        location : string
            Location code, e.g. '00'.
        channel : string
            Channel code, e.g. 'EHE'.
        starttime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            Start date and time.
        endtime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            End date and time.
        quality : 'D', 'R', 'Q', 'M' or 'B', optional
            MiniSEED data quality indicator. M and B (default) are treated the
            same and indicate best available. If M or B are selected, the
            output data records will be stamped with a M.

        Returns
        -------
            :class:`~obspy.core.stream.Stream`
        """
        # convert UTCDateTime to string for query
        try:
            kwargs['starttime'] = \
                    UTCDateTime(kwargs['starttime']).formatIRISWebService()
        except KeyError:
            pass
        try:
            kwargs['endtime'] = \
                    UTCDateTime(kwargs['endtime']).formatIRISWebService()
        except KeyError:
            pass
        # build up query
        url = '/dataselect/query'
        try:
            data = self._fetch(url, **kwargs)
        except HTTPError:
            raise Exception("No waveform data available")
        # create temporary file for writing data
        tf = NamedTemporaryFile()
        tf.write(data)
        # read stream using obspy.mseed
        tf.seek(0)
        try:
            stream = read(tf.name, 'MSEED')
        except:
            stream = Stream()
        tf.close()
        # remove temporary file:
        try:
            os.remove(tf.name)
        except:
            pass
        return stream

    def bulkdataselect(self, bulk, quality=""):
        """
        Interface for `bulkdataselect`-webservice of IRIS
        (http://www.iris.edu/ws/bulkdataselect/).

        Simple requests with wildcards can be performed via
        :meth:`~obspy.iris.client.Client.getWaveform`. The list with channels
        can also be generated using :meth:`~obspy.iris.client.Client.availability`.

        Example
        -------
        
        >>> from obspy.iris import Client
        >>> from obspy.core import UTCDateTime
        >>> client = Client()

        >>> requests = []
        >>> requests.append("TA A25A -- BHZ 2010-084T00:00:00 2010-084T00:10:00")
        >>> requests.append("TA A25A -- BHN 2010-084T00:00:00 2010-084T00:10:00")
        >>> requests.append("TA A25A -- BHE 2010-084T00:00:00 2010-084T00:10:00")
        >>> requests = "\\n".join(requests) # use only a single backslash!
        >>> print requests
        TA A25A -- BHZ 2010-084T00:00:00 2010-084T00:10:00
        TA A25A -- BHN 2010-084T00:00:00 2010-084T00:10:00
        TA A25A -- BHE 2010-084T00:00:00 2010-084T00:10:00

        >>> st = client.bulkdataselect(requests)
        >>> print st
        3 Trace(s) in Stream:
        TA.A25A..BHE | 2010-03-25T00:00:00.000000Z - 2010-03-25T00:10:00.000000Z | 40.0 Hz, 24001 samples
        TA.A25A..BHN | 2010-03-25T00:00:00.000000Z - 2010-03-25T00:10:00.000000Z | 40.0 Hz, 24001 samples
        TA.A25A..BHZ | 2010-03-25T00:00:00.000000Z - 2010-03-25T00:10:00.000000Z | 40.0 Hz, 24001 samples
        
        Parameters
        ----------
        bulk : string
            List of channels to fetch as returned by
            :meth:`~obspy.iris.client.Client.availability`.
            Can be a filename with a text file in bulkdataselect compatible
            format or a string in the same format.
        quality : 'D', 'R', 'Q', 'M' or 'B', optional
            MiniSEED data quality indicator. M and B (default) are treated the
            same and indicate best available. If M or B are selected, the
            output data records will be stamped with a M.

        Returns
        -------
            :class:`~obspy.core.stream.Stream`
        """
        url = '/bulkdataselect/query'
        # check for file
        if os.path.isfile(bulk):
            bulk = open(bulk).read()
        # quality parameter is optional
        if quality:
            bulk = "quality %s\n" % quality.upper() + bulk
        # build up query
        try:
            data = self._HTTP_request(url, data=bulk)
        except HTTPError:
            raise Exception("No waveform data available")
        # create temporary file for writing data
        tf = NamedTemporaryFile()
        tf.write(data)
        # read stream using obspy.mseed
        tf.seek(0)
        try:
            stream = read(tf.name, 'MSEED')
        except:
            stream = Stream()
        tf.close()
        # remove temporary file:
        try:
            os.remove(tf.name)
        except:
            pass
        return stream

    def availability(self, network="*", station="*", location="*", channel="*",
                      starttime=UTCDateTime() - (60 * 60 * 24 * 7), endtime=UTCDateTime() - (60 * 60 * 24 * 7) + 10, output="bulk"):
        """
        Interface for `availability`-webservice of IRIS
        (http://www.iris.edu/ws/availability/).
        Returns list of available channels that can be requested using the
        `bulkdataselect`-webservice.

        Example
        -------
        
        >>> from obspy.iris import Client
        >>> from obspy.core import UTCDateTime
        >>> client = Client()
           
        >>> t1 = UTCDateTime("2010-02-27T06:30:00")
        >>> t2 = UTCDateTime("2010-02-27T06:40:00")
        >>> response = client.availability(network="IU", station="B*",
        ...         channel="BH*", starttime=t1, endtime=t2)
        >>> print response
        IU BBSR 00 BH1 2010-02-27T06:30:00 2010-02-27T06:40:00
        IU BBSR 00 BH2 2010-02-27T06:30:00 2010-02-27T06:40:00
        IU BBSR 00 BHZ 2010-02-27T06:30:00 2010-02-27T06:40:00
        IU BBSR 10 BHE 2010-02-27T06:30:00 2010-02-27T06:40:00
        IU BBSR 10 BHN 2010-02-27T06:30:00 2010-02-27T06:40:00
        IU BBSR 10 BHZ 2010-02-27T06:30:00 2010-02-27T06:40:00
        IU BILL 00 BHE 2010-02-27T06:30:00 2010-02-27T06:40:00
        IU BILL 00 BHN 2010-02-27T06:30:00 2010-02-27T06:40:00
        IU BILL 00 BHZ 2010-02-27T06:30:00 2010-02-27T06:40:00
        <BLANKLINE>

        >>> st = client.bulkdataselect(response)
        >>> print st
        9 Trace(s) in Stream:
        IU.BBSR.00.BH1 | 2010-02-27T06:30:00.019536Z - 2010-02-27T06:39:59.994536Z | 40.0 Hz, 24000 samples
        IU.BBSR.00.BH2 | 2010-02-27T06:30:00.019538Z - 2010-02-27T06:39:59.994538Z | 40.0 Hz, 24000 samples
        IU.BBSR.00.BHZ | 2010-02-27T06:30:00.019538Z - 2010-02-27T06:39:59.994538Z | 40.0 Hz, 24000 samples
        IU.BBSR.10.BHE | 2010-02-27T06:30:00.019538Z - 2010-02-27T06:39:59.994538Z | 40.0 Hz, 24000 samples
        IU.BBSR.10.BHN | 2010-02-27T06:30:00.019538Z - 2010-02-27T06:39:59.994538Z | 40.0 Hz, 24000 samples
        IU.BBSR.10.BHZ | 2010-02-27T06:30:00.019538Z - 2010-02-27T06:39:59.994538Z | 40.0 Hz, 24000 samples
        IU.BILL.00.BHE | 2010-02-27T06:30:00.036324Z - 2010-02-27T06:39:59.986324Z | 20.0 Hz, 12000 samples
        IU.BILL.00.BHN | 2010-02-27T06:30:00.036324Z - 2010-02-27T06:39:59.986324Z | 20.0 Hz, 12000 samples
        IU.BILL.00.BHZ | 2010-02-27T06:30:00.036324Z - 2010-02-27T06:39:59.986324Z | 20.0 Hz, 12000 samples

        Parameters
        ----------
        network : string
            Network code, e.g. 'BW', wildcards allowed.
        station : string
            Station code, e.g. 'MANZ', wildcards allowed.
        location : string
            Location code, e.g. '00', wildcards allowed.
        channel : string
            Channel code, e.g. 'EHE', wildcards allowed.
        starttime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            Start date and time.
        endtime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            End date and time.
        output : string
            Either "bulk" or "xml".

        Returns
        -------
            String that lists available channels, either as plaintext
            `bulkdataselect` format (`output="bulk"`) or in xml format
            (`output="xml"`).
        """
        url = '/availability/query'
        # build up query
        kwargs = {}
        kwargs['network'] = str(network)
        kwargs['station'] = str(station)
        kwargs['location'] = str(location)
        kwargs['channel'] = str(channel)
        try:
            kwargs['starttime'] = UTCDateTime(starttime).formatIRISWebService()
        except:
            kwargs['starttime'] = starttime
        try:
            kwargs['endtime'] = UTCDateTime(endtime).formatIRISWebService()
        except:
            kwargs['endtime'] = endtime
        kwargs['output'] = str(output)
        if not kwargs['output'] in ("bulk", "xml"):
            msg = "kwarg output must be either 'bulk' or 'xml'."
            raise ValueError(msg)
        data = self._fetch(url, **kwargs)
        return data


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
