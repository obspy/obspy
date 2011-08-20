# -*- coding: utf-8 -*-
"""
IRIS Web service client for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from obspy.core import UTCDateTime, read, Stream
from obspy.core.util import NamedTemporaryFile, BAND_CODE, _getVersionString
from urllib2 import HTTPError
import os
import platform
import sys
import urllib
import urllib2
try: # pragma: no cover
    import json
    if not getattr(json, "loads", None):
        json.loads = json.read #@UndefinedVariable
except ImportError: # pragma: no cover
    import simplejson as json


VERSION = _getVersionString("obspy.iris")
DEFAULT_USER_AGENT = "ObsPy %s (%s, Python %s)" % (VERSION, platform.platform(),
                                                   platform.python_version())


class Client(object):
    """
    IRIS Web service request client.

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
                 user="", password="", timeout=10, debug=False,
                 user_agent=DEFAULT_USER_AGENT):
        """
        Initializes the IRIS Web service client.

        Parameters
        ----------
        base_url : string, optional
            Base URL of the IRIS Web service (default is
            'http://www.iris.edu/ws').
        user : string, optional
            The user name used for authentication with the Web service (default
            an empty string).
        password : string, optional
            A password used for authentication with the Web service (default is
            an empty string).
        timeout : int, optional
            Seconds before a connection timeout is raised (default is 10
            seconds). This works only for Python >= 2.6.x.
        debug : boolean, optional
            Enables verbose output (default is False).
        user_agent : string, optional
            Sets an client identification string which may be used on server
            side for statistical analysis (default contains the current module
            version and basic information about the used operation system, e.g.
            'ObsPy 0.4.7.dev-r2432 (Windows-7-6.1.7601-SP1, Python 2.7.1)'.
        """
        self.base_url = base_url
        self.timeout = timeout
        self.debug = debug
        self.user_agent = user_agent
        # Create an OpenerDirector for Basic HTTP Authentication
        password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, base_url, user, password)
        auth_handler = urllib2.HTTPBasicAuthHandler(password_mgr)
        opener = urllib2.build_opener(auth_handler)
        # install globally
        urllib2.install_opener(opener)

    def _fetch(self, url, data=None, headers={}, **params):
        """
        Send a HTTP request via urllib2.

        :type url: String
        :param url: Complete URL of resource
        :type data: String
        :param data: Channel list as returned by `availability` Web service
        :type headers: dict
        :param headers: Additional header information for request
        """
        headers['User-Agent'] = self.user_agent
        # replace special characters 
        remoteaddr = self.base_url + url
        if params:
            remoteaddr += '?' + urllib.urlencode(params)
        if self.debug:
            print('\nRequesting %s' % (remoteaddr))
        req = urllib2.Request(url=remoteaddr, data=data, headers=headers)
        # timeout exists only for Python >= 2.6
        if sys.hexversion < 0x02060000: # pragma: no cover
            response = urllib2.urlopen(req)
        else: # pragma: no cover
            response = urllib2.urlopen(req, timeout=self.timeout)
        doc = response.read()
        return doc

    def getWaveform(self, network, station, location, channel, starttime,
                     endtime, quality='B'):
        """
        Gets a ObsPy Stream object.
        Wildcards are allowed for `network`, `station`, `location` and
        `channel`.

        Examples
        --------
        (1) Requesting waveform of a single channel.

            >>> from obspy.iris import Client
            >>> from obspy.core import UTCDateTime
            >>> client = Client()
            >>> t1 = UTCDateTime("2010-02-27T06:30:00.000")
            >>> t2 = UTCDateTime("2010-02-27T07:00:00.000")
            >>> st = client.getWaveform("IU", "ANMO", "00", "BHZ", t1, t2)
            >>> print(st)
            1 Trace(s) in Stream:
            IU.ANMO.00.BHZ | 2010-02-27T06:30:00.019538Z - 2010-02-27T07:00:00.019538Z | 20.0 Hz, 36001 samples

        (2) Requesting waveforms of multiple channels using wildcard characters.

            >>> t1 = UTCDateTime("2010-084T00:00:00")
            >>> t2 = UTCDateTime("2010-084T00:30:00")
            >>> st = client.getWaveform("TA", "A25A", "", "BH*", t1, t2)
            >>> print(st)
            3 Trace(s) in Stream:
            TA.A25A..BHE | 2010-03-25T00:00:00.000001Z - 2010-03-25T00:30:00.000001Z | 40.0 Hz, 72001 samples
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
        # try to be intelligent in starttime/endtime extension for fetching
        # data
        try:
            t_extension = 2.0 / BAND_CODE[kwargs['channel'][0]]
        except:
            # use 1 second extension if no proper bandcode info
            t_extension = 1.0
        kwargs['starttime'] = UTCDateTime(starttime) - t_extension
        kwargs['endtime'] = UTCDateTime(endtime) + t_extension
        if str(quality).upper() in ['D', 'R', 'Q', 'M', 'B']:
            kwargs['quality'] = str(quality).upper()

        # single channel request, go via `dataselect` Web service
        if all([val.isalnum() for val in (kwargs['network'],
                                          kwargs['station'],
                                          kwargs['location'],
                                          kwargs['channel'])]):
            st = self.dataselect(**kwargs)
        # wildcarded channel request, go via `availability` and
        # `bulkdataselect` Web services
        else:
            quality = kwargs.pop("quality", "")
            bulk = self.availability(**kwargs)
            st = self.bulkdataselect(bulk, quality)
        st.trim(UTCDateTime(starttime), UTCDateTime(endtime))
        return st

    def saveWaveform(self, filename, network, station, location, channel,
                     starttime, endtime, quality='B'):
        """
        Writes a retrieved waveform directly into a file.

        This method ensures the storage of the unmodified waveform data
        delivered by the IRIS server, e.g. preserving the record based
        quality flags of MiniSEED files which would be neglected reading it
        with obspy.mseed.

        Example
        -------
        >>> from obspy.iris import Client
        >>> from obspy.core import UTCDateTime
        >>> client = Client()
        >>> t1 = UTCDateTime('2010-02-27T06:30:00.000')
        >>> t2 = UTCDateTime('2010-02-27T10:30:00.000')
        >>> client.saveWaveform('IU.ANMO.00.BHZ.mseed', 'IU', 'ANMO',
        ...                     '00', 'BHZ', t1, t2) # doctest: +SKIP

        Parameters
        ----------
        filename : string
            Name of the output file.
        network : string
            Network code, e.g. 'IU'.
        station : string
            Station code, e.g. 'ANMO'.
        location : string
            Location code, e.g. '00'.
        channel : string
            Channel code, e.g. 'BHZ'.
        starttime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            Start date and time.
        endtime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            End date and time.
        quality : 'D', 'R', 'Q', 'M' or 'B', optional
            MiniSEED data quality indicator. M and B (default) are treated the
            same and indicate best available. If M or B are selected, the
            output data records will be stamped with a M.
        """
        kwargs = {}
        kwargs['network'] = str(network)[0:2]
        kwargs['station'] = str(station)[0:5]
        if location:
            kwargs['location'] = str(location)[0:2]
        else:
            kwargs['location'] = '--'
        kwargs['channel'] = str(channel)[0:3]
        kwargs['filename'] = str(filename)
        # try to be intelligent in starttime/endtime extension for fetching
        # data
        try:
            t_extension = 2.0 / BAND_CODE[kwargs['channel'][0]]
        except:
            # use 1 second extension if no proper bandcode info
            t_extension = 1.0
        kwargs['starttime'] = UTCDateTime(starttime) - t_extension
        kwargs['endtime'] = UTCDateTime(endtime) + t_extension
        if str(quality).upper() in ['D', 'R', 'Q', 'M', 'B']:
            kwargs['quality'] = str(quality).upper()
        self.dataselect(**kwargs)

    def saveResponse(self, filename, network, station, location, channel,
                     starttime, endtime, format='RESP'):
        """
        Writes a response information into a file. Possible output formats are
        RESP (http://www.iris.edu/KB/questions/69/What+is+a+RESP+file%3F) 
        or StationXML (http://www.data.scec.org/xml/station/).

        Parameters
        ----------
        filename : string
            Name of the output file.
        network : string
            Network code, e.g. 'IU'.
        station : string
            Station code, e.g. 'ANMO'.
        location : string
            Location code, e.g. '00' or '*'.
        channel : string
            Channel code, e.g. 'BHZ' or 'B*'.
        starttime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            Start date and time.
        endtime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            End date and time.
        format : 'RESP' or 'StationXML', optional
            Output format.
        """
        kwargs = {}
        kwargs['network'] = str(network)[0:2]
        kwargs['station'] = str(station)[0:5]
        if location:
            kwargs['location'] = str(location)[0:2]
        else:
            kwargs['location'] = '--'
        kwargs['channel'] = str(channel)[0:3]
        kwargs['starttime'] = UTCDateTime(starttime)
        kwargs['endtime'] = UTCDateTime(endtime)
        if format == 'StationXML':
            data = self.station(level='resp', **kwargs)
        else:
            data = self.resp(**kwargs)
        fh = open(filename, "wb")
        fh.write(data)
        fh.close()

    def resp(self, **kwargs):
        """
        Interface for `resp` Web service of IRIS (http://www.iris.edu/ws/resp/).

        This method provides access to channel response information in the SEED
        RESP (http://www.iris.edu/KB/questions/69/What+is+a+RESP+file%3F) format
        (as used by evalresp). Users can query for channel response by network,
        station, channel, location and time. 

        Example
        -------
        >>> from obspy.iris import Client
        >>> from obspy.core import UTCDateTime
        >>> client = Client()
        >>> t1 = UTCDateTime("2010-02-27T06:30:00.000")
        >>> t2 = UTCDateTime("2010-02-27T10:30:00.000")
        >>> data = client.resp(network="IU", station="ANMO", location="00",
        ...                    channel="BHZ", starttime=t1, endtime=t2)
        >>> print(data)  # doctest: +ELLIPSIS
        #
        ###################################################################################
        #
        B050F03     Station:     ANMO
        B050F16     Network:     IU
        B052F03     Location:    00
        B052F04     Channel:     BHZ
        ...

        Parameters
        ----------
        network : string
            Network code, e.g. 'IU'.
        station : string
            Station code, e.g. 'ANMO'.
        location : string
            Location code, e.g. '00', wildcards allowed.
        channel : string
            Channel code, e.g. 'BHZ', wildcards allowed.
        starttime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            Start date and time.
        endtime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            End date and time.

        Returns
        -------
            SEED RESP file as string.
            
        .. _GEOFON:
            http://geofon.gfz-potsdam.de
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
        url = '/resp/query'
        try:
            data = self._fetch(url, **kwargs)
        except HTTPError, e:
            msg = "No response data available (%s: %s)"
            msg = msg % (e.__class__.__name__, e)
            raise Exception(msg)
        return data

    def station(self, **kwargs):
        """
        Interface for `station` Web service of IRIS
        (http://www.iris.edu/ws/station/).

        This method provides access to station metadata in the IRIS DMC
        database. The results are returned in XML format using the StationXML
        schema (http://www.data.scec.org/xml/station/). Users can query for
        station metadata by network, station, channel, location, time and other
        search criteria and request results at multiple levels (station,
        channel, response, etc.).

        Example
        -------
        >>> from obspy.iris import Client
        >>> from obspy.core import UTCDateTime
        >>> client = Client()
        >>> t1 = UTCDateTime("2006-03-01")
        >>> t2 = UTCDateTime("2006-09-01")
        >>> station_xml = client.station(network="IU", station="ANMO",
        ...                              location="00", channel="BHZ",
        ...                              starttime=t1, endtime=t2, level="net")
        >>> print(station_xml) # doctest: +ELLIPSIS
        <BLANKLINE>
        <StaMessage xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns="http://www.data.scec.org/xml/station/" xsi:schemaLocation="http://www.data.scec.org/xml/station/ http://www.data.scec.org/xml/station/station.xsd">
         <Source>IRIS-DMC</Source>
         <Sender>IRIS-DMC</Sender>
         <Module>IRIS WEB SERVICE: http://www.iris.edu/ws/station</Module>
         <ModuleURI>http://www.iris.edu/ws/station/query?network=IU&amp;level=net&amp;timewindow=2006-03-01%2C2006-09-01&amp;station=ANMO&amp;location=00&amp;channel=BHZ</ModuleURI>
         <SentDate>...</SentDate>
         <Network net_code="IU">
          <StartDate>1988-01-01T00:00:00</StartDate>
          <EndDate>2500-12-12T23:59:59</EndDate>
          <Description>Global Seismograph Network (GSN - IRIS/USGS)</Description>
          <TotalNumberStations>91</TotalNumberStations>
         </Network>
        </StaMessage>

        Parameters
        ----------
        network : string
            Network code, e.g. 'IU'.
        station : string
            Station code, e.g. 'ANMO'.
        location : string
            Location code, e.g. '00', wildcards allowed.
        channel : string
            Channel code, e.g. 'BHZ', wildcards allowed.
        starttime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            Start date. Hours, minutes and seconds are ignored.
        endtime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            End date. Hours, minutes and seconds are ignored.
        level : 'net', 'sta', 'chan', or 'resp', optional
            Specify whether to include channel/response metadata or not.

        Returns
        -------
            StationXML file as string.
        """
        # convert UTCDateTime to string for query
        try:
            starttime = UTCDateTime(kwargs['starttime']).date
        except KeyError:
            starttime = UTCDateTime(1900, 1, 1)
        try:
            endtime = UTCDateTime(kwargs['endtime']).date
        except KeyError:
            endtime = UTCDateTime(2501, 1, 1)
        kwargs.pop('starttime')
        kwargs.pop('endtime')
        kwargs['timewindow'] = '%s,%s' % (starttime, endtime)
        # build up query
        url = '/station/query'
        try:
            data = self._fetch(url, **kwargs)
        except HTTPError, e:
            msg = "No response data available (%s: %s)"
            msg = msg % (e.__class__.__name__, e)
            raise Exception(msg)
        return data

    def dataselect(self, filename=None, **kwargs):
        """
        Interface for `dataselect` Web service of IRIS
        (http://www.iris.edu/ws/dataselect/).

        This method returns a single channel of time series data (no wildcards
        are allowed). With this service you specify network, station, location,
        channel and a time range and the service returns either as an ObsPy
        :class:`~obspy.core.stream.Stream` object or saves the data directly
        as MiniSEED file.

        Example
        -------
        >>> from obspy.iris import Client
        >>> from obspy.core import UTCDateTime
        >>> client = Client()
        >>> t1 = UTCDateTime("2010-02-27T06:30:00.000")
        >>> t2 = UTCDateTime("2010-02-27T07:00:00.000")
        >>> st = client.dataselect(network="IU", station="ANMO", location="00",
        ...                        channel="BHZ", starttime=t1, endtime=t2)
        >>> print(st)
        1 Trace(s) in Stream:
        IU.ANMO.00.BHZ | 2010-02-27T06:30:00.019538Z - 2010-02-27T06:59:59.969538Z | 20.0 Hz, 36000 samples

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
        filename : string, optional
            Name of a output file. If this parameter is given nothing will be
            returned. Default is None.

        Returns
        -------
            :class:`~obspy.core.stream.Stream` if no `filename` is given.
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
        except HTTPError, e:
            msg = "No waveform data available (%s: %s)"
            msg = msg % (e.__class__.__name__, e)
            raise Exception(msg)
        # if filename is given, create fh, write to file and return
        if filename:
            if isinstance(filename, basestring):
                fh = open(filename, "wb")
            elif isinstance(filename, file):
                fh = filename
            else:
                msg = "Parameter filename must be either string or file " + \
                    "handler."
                raise TypeError(msg)
            fh.write(data)
            fh.close()
            return
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
        Interface for `bulkdataselect` Web service of IRIS
        (http://www.iris.edu/ws/bulkdataselect/).

        This method returns multiple channels of time series data for specified
        time ranges. With this service you specify a list of selections composed
        of network, station, location, channel, starttime and endtime and the
        service streams back the selected raw waveform data as an ObsPy
        :class:`~obspy.core.stream.Stream` object.

        Simple requests with wildcards can be performed via
        :meth:`~obspy.iris.client.Client.getWaveform`. The list with channels
        can also be generated using
        :meth:`~obspy.iris.client.Client.availability`.

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
        >>> print(requests)
        TA A25A -- BHZ 2010-084T00:00:00 2010-084T00:10:00
        TA A25A -- BHN 2010-084T00:00:00 2010-084T00:10:00
        TA A25A -- BHE 2010-084T00:00:00 2010-084T00:10:00

        >>> st = client.bulkdataselect(requests)
        >>> print(st)
        3 Trace(s) in Stream:
        TA.A25A..BHE | 2010-03-25T00:00:00.000001Z - 2010-03-25T00:10:00.000001Z | 40.0 Hz, 24001 samples
        TA.A25A..BHN | 2010-03-25T00:00:00.000000Z - 2010-03-25T00:10:00.000000Z | 40.0 Hz, 24001 samples
        TA.A25A..BHZ | 2010-03-25T00:00:00.000000Z - 2010-03-25T00:09:59.975000Z | 40.0 Hz, 24000 samples

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
            data = self._fetch(url, data=bulk)
        except HTTPError, e:
            msg = "No waveform data available (%s: %s)"
            msg = msg % (e.__class__.__name__, e)
            raise Exception(msg)
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

    def availability(self, network="*", station="*", location="*",
                     channel="*", starttime=UTCDateTime() - (60 * 60 * 24 * 7),
                     endtime=UTCDateTime() - (60 * 60 * 24 * 7) + 10,
                     lat=None, lon=None, minradius=None, maxradius=None,
                     minlat=None, maxlat=None, minlon=None, maxlon=None,
                     output="bulk"):
        """
        Interface for `availability` Web service of IRIS
        (http://www.iris.edu/ws/availability/).

        This method returns information about what time series data is available
        at the IRIS-DMC. Users can query for station metadata by network,
        station, channel, location, time and other search criteria. Results are
        may be formated in two formats: 'bulk' or 'xml'.

        The 'bulk' formatted information can be passed directly to the
        :meth:`~obspy.iris.client.Client.bulkdataselect()` method.

        The XML format contains station locations as well as channel time range
        for :meth:`~obspy.iris.client.Client.availability()`.

        Example
        -------
        >>> from obspy.iris import Client
        >>> from obspy.core import UTCDateTime
        >>> client = Client()
        >>> t1 = UTCDateTime("2010-02-27T06:30:00")
        >>> t2 = UTCDateTime("2010-02-27T06:40:00")
        >>> response = client.availability(network="IU", station="B*",
        ...         channel="BH*", starttime=t1, endtime=t2)
        >>> print(response)
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
        >>> print(st)
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
        lat : float or int (or string)
            Latitude of center point for circular bounding area.
        lon : float or int (or string)
            Longitude of center point for circular bounding area.
        minradius : float or int (or string)
            Minimum radius for circular bounding area.
        maxradius : float or int (or string)
            Maximum radius for circular bounding area.
        minlat : float or int (or string)
            Minimum latitude for rectangular bounding box.
        minlon : float or int (or string)
            Minimum longitude for rectangular bounding box.
        maxlat : float or int (or string)
            Maximum latitude for rectangular bounding box.
        maxlon : float or int (or string)
            Maximum longitude for rectangular bounding box.
        output : string
            Either "bulk" or "xml".

        :note: Restricting returned data by geographical coordinates either
               *all* of `minlat`, `maxlat`, `minlon` and `maxlon` have to be
               specified (for a rectangular bounding box) _or_ *all* of `lat`,
               `lon`, `minradius` and `maxradius` have to be specified (for a
               circular bounding area)

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
        # sanity checking geographical bounding areas
        rectangular = (minlat, minlon, maxlat, maxlon)
        circular = (lon, lat, minradius, maxradius)
        # not both can be specified at the same time
        if any(rectangular) and any(circular):
            msg = "Rectangular and circular bounding areas can not be combined"
            raise ValueError(msg)
        # check and setup rectangular box criteria
        if any(rectangular):
            if not all(rectangular):
                msg = "Missing constraints for rectangular bounding box"
                raise ValueError(msg)
            kwargs['minlat'] = str(minlat)
            kwargs['minlon'] = str(minlon)
            kwargs['maxlat'] = str(maxlat)
            kwargs['maxlon'] = str(maxlon)
        # check and setup circular box criteria
        if any(circular):
            if not all(circular):
                msg = "Missing constraints for circular bounding area"
                raise ValueError(msg)
            kwargs['lat'] = str(lat)
            kwargs['lon'] = str(lon)
            kwargs['minradius'] = str(minradius)
            kwargs['maxradius'] = str(maxradius)
        # checking output options
        if not kwargs['output'] in ("bulk", "xml"):
            msg = "kwarg output must be either 'bulk' or 'xml'."
            raise ValueError(msg)
        data = self._fetch(url, **kwargs)
        return data

    def sacpz(self, network="*", station="*", location="*", channel="*",
              starttime=UTCDateTime() - (60 * 60 * 24 * 7),
              endtime=UTCDateTime() - (60 * 60 * 24 * 7) + 10,
              output="sacpz"):
        """
        Interface for `sacpz` Web service of IRIS
        (http://www.iris.edu/ws/sacpz/).

        This method provides access to instrument response information
        (per-channel) as poles and zeros in the ASCII format used by SAC and
        other programs. Users can query for channel response by network,
        station, channel, location and time.

        Example
        -------
        >>> from obspy.iris import Client
        >>> from obspy.core import UTCDateTime
        >>> client = Client()
        >>> t1 = UTCDateTime("2005-01-01")
        >>> t2 = UTCDateTime("2008-01-01")
        >>> sacpz = client.sacpz(network="IU", station="ANMO", location="00",
        ...                      channel="BHZ", starttime=t1, endtime=t2)
        >>> print(sacpz) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        * **********************************
        * NETWORK   (KNETWK): IU
        * STATION    (KSTNM): ANMO
        * LOCATION   (KHOLE): 00
        * CHANNEL   (KCMPNM): BHZ
        * CREATED           : ...
        * START             : 2002-11-19T21:07:00
        * END               : 2008-06-30T00:00:00
        * DESCRIPTION       : Albuquerque, New Mexico, USA
        * LATITUDE          : 34.945981
        * LONGITUDE         : -106.457133
        * ELEVATION         : 1671.0
        * DEPTH             : 145.0
        * DIP               : 0.0
        * AZIMUTH           : 0.0
        * SAMPLE RATE       : 20.0
        * INPUT UNIT        : M
        * OUTPUT UNIT       : COUNTS
        * INSTTYPE          : Geotech KS-54000 Borehole Seismometer
        * INSTGAIN          : 2.204000e+03 (M/S)
        * COMMENT           : 
        * SENSITIVITY       : 9.244000e+08 (M/S)
        * A0                : 8.608300e+04
        * **********************************
        ZEROS    3
            +0.000000e+00    +0.000000e+00    
            +0.000000e+00    +0.000000e+00    
            +0.000000e+00    +0.000000e+00    
        POLES    5
            -5.943130e+01    +0.000000e+00    
            -2.271210e+01    +2.710650e+01    
            -2.271210e+01    -2.710650e+01    
            -4.800400e-03    +0.000000e+00    
            -7.319900e-02    +0.000000e+00    
        CONSTANT    7.957513e+13
        <BLANKLINE>
        <BLANKLINE>
        <BLANKLINE>

        Parameters
        ----------
        network : string
            Network code, e.g. 'IU', wildcards allowed.
        station : string
            Station code, e.g. 'ANMO', wildcards allowed.
        location : string
            Location code, e.g. '00', wildcards allowed.
        channel : string
            Channel code, e.g. 'BHZ', wildcards allowed.
        starttime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            Start date and time.
        endtime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            End date and time.
        output : string
            Currently only "sacpz".

        Returns
        -------
            String with SAC poles and zeros information.
        """
        url = '/sacpz/query'
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
        data = self._fetch(url, **kwargs)
        # check output options
        kwargs['output'] = str(output)
        if kwargs['output'] == "sacpz":
            return data
        else:
            msg = "unsupported output option: %s" % kwargs['output']
            raise ValueError(msg)

    def distaz(self, **kwargs):
        """
        Interface for `distaz` Web service of IRIS
        (http://www.iris.edu/ws/distaz/).

        This method will calculate the distance and azimuth between two points
        on a sphere. Azimuths start at north and go clockwise, i.e.
        0/360 = north, 90 = east, 180 = south, 270 = west. The azimuth is from
        the station to the event, the backazimuth is from the event to the
        station.

        Example
        -------
        >>> from obspy.iris import Client
        >>> client = Client()
        >>> result = client.distaz(stalat=1.1, stalon=1.2, evtlat=3.2, 
        ...                        evtlon=1.4)
        >>> print(result)
        {'distance': 2.09554, 'backazimuth': 5.46946, 'azimuth': 185.47692}

        Parameters
        ----------
        stalat : float
            Station latitude.
        stalon : float
            Station longitude.
        evtlat : float
            Event latitude.
        evtlon : float
            Event longitude.

        Returns
        -------
            Dictionary containing values for azimuth, backazimuth and distance.
        """
        # set JSON as expected content type
        headers = {'Accept': 'application/json'}
        # build up query
        url = '/distaz/query'
        try:
            data = self._fetch(url, headers=headers, **kwargs)
        except HTTPError, e:
            msg = "No response data available (%s: %s)"
            msg = msg % (e.__class__.__name__, e)
            raise Exception(msg)
        data = json.loads(data)
        results = {}
        results['distance'] = data['DistanceAzimuth']['distance']
        results['backazimuth'] = data['DistanceAzimuth']['backAzimuth']
        results['azimuth'] = data['DistanceAzimuth']['azimuth']
        return results

    def flinnengdahl(self, lat, lon, rtype="both"):
        """
        Interface for `flinnengdahl` Web service of IRIS
        (http://www.iris.edu/ws/flinnengdahl/).

        This method converts a latitude, longitude pair into either a Flinn-
        Engdahl region code or name.

        Example
        -------
        >>> from obspy.iris import Client
        >>> client = Client()
        >>> client.flinnengdahl(lat=-20.5, lon=-100.6, rtype="code")
        683
        >>> client.flinnengdahl(lat=42, lon=-122.24, rtype="region")
        'OREGON'
        >>> client.flinnengdahl(lat=-20.5, lon=-100.6)
        (683, 'SOUTHEAST CENTRAL PACIFIC OCEAN')

        :param lat: Latitude of interest.
        :type lat: float
        :param lon: Longitude of interest.
        :type lon: float
        :param rtype: 'code', 'region' or 'both' (default) 
        :type rtype: 

        :rtype: int, string, or (int, string)
        :returns: Returns Flinn-Engdahl region code or name or both, depending
            on the request type parameter ``rtype``.
        """
        url = '/flinnengdahl/%s?lat=%f&lon=%f'
        # check rtype
        try:
            if rtype == 'code':
                return int(self._fetch(url % (rtype, lat, lon)))
            elif rtype == 'region':
                return self._fetch(url % (rtype, lat, lon)).strip()
            else:
                code = int(self._fetch(url % ('code', lat, lon)))
                region = self._fetch(url % ('region', lat, lon)).strip()
                return (code, region)
        except HTTPError, e:
            msg = "No Flinn-Engdahl data available (%s: %s)"
            msg = msg % (e.__class__.__name__, e)
            raise Exception(msg)


if __name__ == '__main__': # pragma: no cover
    import doctest
    doctest.testmod(exclude_empty=True)
