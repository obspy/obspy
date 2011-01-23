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
    >>>
    >>> t = UTCDateTime("2010-02-27T06:30:00.000")
    >>> client = Client()
    >>>
    >>> st = client.getWaveform("IU", "ANMO", "00", "BHZ", t, t + 20)
    >>> print st
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
        kwargs['starttime'] = (UTCDateTime(starttime) - t_extension).formatIRISWebService()
        kwargs['endtime'] = (UTCDateTime(endtime) + t_extension).formatIRISWebService()
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
        url = '/dataselect/query'
        # build up query
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

        Parameters
        ----------
        bulk : string
            List of channels to fetch as returned by :meth:`~obspy.iris.client.Client.availability`
        quality : 'D', 'R', 'Q', 'M' or 'B', optional
            MiniSEED data quality indicator. M and B (default) are treated the
            same and indicate best available. If M or B are selected, the
            output data records will be stamped with a M.

        Returns
        -------
            :class:`~obspy.core.stream.Stream`
        """
        url = '/bulkdataselect/query'
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
                      starttime=UTCDateTime()-3600, endtime=UTCDateTime()-3700, output="bulk"):
        """
        Interface for `availability`-webservice of IRIS
        (http://www.iris.edu/ws/availability/).
        Returns list of available channels that can be requested using the
        `bulkdataselect`-webservice.

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
