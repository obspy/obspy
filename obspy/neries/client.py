# -*- coding: utf-8 -*-
"""
NERIES Web service client for ObsPy.

.. seealso:: http://www.seismicportal.eu/jetspeed/portal/web-services.psml

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future import standard_library
from future.utils import native_str

import functools
import json
import platform
import warnings

with standard_library.hooks():
    import urllib.parse
    import urllib.request

from suds.client import Client as SudsClient
from suds.plugin import MessagePlugin
from suds.sax.attribute import Attribute
from suds.xsd.sxbase import SchemaObject

from obspy import Stream, UTCDateTime, __version__, read
from obspy.core.util import NamedTemporaryFile, guessDelta


DEPR_WARN = ("This service was shut down on the server side, please use the "
             "obspy.fdsn Client instead (with `base_url='NERIES'`).")

SEISMOLINK_WSDL = "http://www.orfeus-eu.org/wsdl/seismolink/seismolink.wsdl"
TAUP_WSDL = "http://www.orfeus-eu.org/wsdl/taup/taup.wsdl"

MAP = {'min_datetime': "dateMin", 'max_datetime': "dateMax",
       'min_latitude': "latMin", 'max_latitude': "latMax",
       'min_longitude': "lonMin", 'max_longitude': "lonMax",
       'min_depth': "depthMin", 'max_depth': "depthMax",
       'min_magnitude': "magMin", 'max_magnitude': "magMax",
       'magnitude_type': "magType", 'author': "auth",
       'max_results': "limit", 'sort_by': "sort", 'sort_direction': "dir",
       'format': "format", 'datetime': "datetime", 'depth': "depth",
       'flynn_region': "flynn_region", 'latitude': "lat",
       'longitude': "lon", 'magnitude': "mag", 'origin_id': "orid",
       'event_id': "unid"}

MAP_INVERSE = dict([(value, key) for key, value in MAP.items()])
# in results the "magType" key is all lowercase, so add it to..
MAP_INVERSE['magtype'] = "magnitude_type"

DEFAULT_USER_AGENT = "ObsPy %s (%s, Python %s)" % (__version__,
                                                   platform.platform(),
                                                   platform.python_version())
MAX_REQUESTS = 50


# monkey patching SUDS
# ses also https://fedorahosted.org/suds/ticket/292


def _namespace(self, prefix=None):
    if self.ref is not None:
        return ('', self.ref[1])
    ns = self.schema.tns
    if ns[0] is None:
        ns = (prefix, ns[1])
    return ns

SchemaObject.namespace = _namespace


def _mapKwargs(f):
    """
    Maps function arguments to keyword arguments.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        # set some default values
        new_kwargs = {'sort': "datetime", 'dir': "ASC", 'limit': 100,
                      'format': "list"}
        for key in kwargs:
            if key in MAP:
                new_kwargs[MAP[key]] = kwargs[key]
        v = f(*args, **new_kwargs)
        return v
    return wrapper


class _AttributePlugin(MessagePlugin):
    """
    Suds plug-in extending the method call with arbitrary attributes.
    """
    def __init__(self, dict):
        self.dict = dict

    def marshalled(self, context):
        method = context.envelope.getChild('Body')[0]
        for key, item in self.dict.items():
            method.attributes.append(Attribute(key, item))


class Client(object):
    """
    NERIES Web service request client.
    """
    def __init__(self, user="", password="", timeout=10, debug=False,
                 user_agent=DEFAULT_USER_AGENT):
        """
        Initializes the NERIES Web service client.

        :type user: str, optional
        :param user: The user name used for identification with the Web
            service. This entry in form of a email address is required for
            using the following methods:

            * :meth:`~obspy.neries.client.Client.saveWaveform`
            * :meth:`~obspy.neries.client.Client.getWaveform`
            * :meth:`~obspy.neries.client.Client.getInventory`

            Defaults to ``''``.
        :type password: str, optional
        :param password: A password used for authentication with the Web
            service. Defaults to ``''``.
        :type timeout: int, optional
        :param timeout: Seconds before a connection timeout is raised (default
            is 10 seconds). Available only for Python >= 2.6.x.
        :type debug: bool, optional
        :param debug: Enables verbose output.
        :type user_agent: str, optional
        :param user_agent: Sets an client identification string which may be
            used on server side for statistical analysis (default contains the
            current module version and basic information about the used
            operation system, e.g.
            ``'ObsPy 0.4.7.dev-r2432 (Windows-7-6.1.7601-SP1, Python 2.7.1)'``.
        """
        self.base_url = "http://www.seismicportal.eu"
        self.timeout = timeout
        self.debug = debug
        self.user_agent = user_agent
        self.user = user
        self.password = password
        # Create an OpenerDirector for Basic HTTP Authentication
        password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, self.base_url, self.user,
                                  self.password)
        auth_handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
        opener = urllib.request.build_opener(auth_handler)
        # install globally
        urllib.request.install_opener(opener)

    def _fetch(self, url, headers={}, **params):
        """
        Send a HTTP request via urllib2.

        :type url: str
        :param url: Complete URL of resource
        :type headers: dict
        :param headers: Additional header information for request
        """
        headers['User-Agent'] = self.user_agent
        # replace special characters
        remoteaddr = self.base_url + url + '?' + \
            urllib.parse.urlencode(params)
        if self.debug:
            print('\nRequesting %s' % (remoteaddr))
        response = urllib.request.urlopen(remoteaddr, timeout=self.timeout)
        doc = response.read()
        return doc

    def _json2list(self, data):
        """
        Converts a JSON formatted string into a event/origin list.
        """
        results = json.loads(data)
        events = []
        float_keys = ('depth', 'latitude', 'longitude', 'magnitude')
        for result in results['unids']:
            event = dict([(MAP_INVERSE[k], v)
                          for k, v in result.items()])
            for k in float_keys:
                event[k] = float(event[k])
            event['magnitude_type'] = event['magnitude_type'].lower()
            event['datetime'] = UTCDateTime(event['datetime'])
            # convention in ObsPy: all depths negative down
            event['depth'] = -event['depth']
            events.append(event)
        return events

    @_mapKwargs
    def getEvents(self, min_datetime=None, max_datetime=None,
                  min_longitude=None, max_longitude=None, min_latitude=None,
                  max_latitude=None, min_depth=None, max_depth=None,
                  min_magnitude=None, max_magnitude=None, magnitude_type=None,
                  author=None, sort_by="datetime", sort_direction="ASC",
                  max_results=100, format=None, **kwargs):
        """
        SHUT DOWN ON SERVER SIDE!

        This service was shut down on the server side, please use the
        obspy.fdsn Client instead (with `base_url='NERIES'`).
        """
        raise Exception(DEPR_WARN)

    def getLatestEvents(self, num=10, format=None):
        """
        SHUT DOWN ON SERVER SIDE!

        This service was shut down on the server side, please use the
        obspy.fdsn Client instead (with `base_url='NERIES'`).
        """
        raise Exception(DEPR_WARN)

    def getEventDetail(self, uri, format=None):
        """
        SHUT DOWN ON SERVER SIDE!

        This service was shut down on the server side, please use the
        obspy.fdsn Client instead (with `base_url='NERIES'`).
        """
        raise Exception(DEPR_WARN)

    def getTravelTimes(self, latitude, longitude, depth, locations=[],
                       model='iasp91'):
        """
        Returns travel times for specified station-event geometry using
        standard velocity models such as ``iasp91``, ``ak135`` or ``qdt``.

        :type latitude: float
        :param latitude: Event latitude.
        :type longitude: float
        :param longitude: Event longitude.
        :type depth: float
        :param depth: Event depth in km.
        :type locations: list of tuples
        :param locations: Each tuple contains a pair of (latitude, longitude)
            of a station.
        :type model: str, optional
        :param model: Velocity model, one of ``'iasp91'``, ``'ak135'``, or
            ``'qdt'``. Defaults to ``'iasp91'``.
        :return: List of dicts containing phase name and arrival times in ms.

        .. seealso:: http://www.orfeus-eu.org/wsdl/taup/taup.wsdl

        .. rubric:: Example

        >>> client = Client()
        >>> locations = [(48.0, 12.0), (48.1, 12.0)]
        >>> result = client.getTravelTimes(latitude=20.0, longitude=20.0,
        ...                                depth=10.0, locations=locations,
        ...                                model='iasp91')
        >>> len(result)
        2
        >>> result[0]  # doctest: +SKIP
        {'P': 356981.13561726053, 'S': 646841.5619481194}
        """
        # enable logging if debug option is set
        if self.debug:
            import logging
            logging.basicConfig(level=logging.INFO)
            logging.getLogger('suds.client').setLevel(logging.DEBUG)
        # initialize client
        client = SudsClient(TAUP_WSDL)
        # set cache of 5 days
        cache = client.options.cache
        cache.setduration(days=5)
        # create request
        request = []
        for location in locations:
            req = {'event-depth': float(depth),
                   'event-lat': float(latitude),
                   'event-lon': float(longitude),
                   'model': str(model),
                   'point-lat': float(location[0]),
                   'point-lon': float(location[1])}
            request.append(req)
        data = client.service.getArrivalTimes(request)
        result = []
        for item in data:
            times = {}
            if hasattr(item, 'arrival-time'):
                for time in item['arrival-time']:
                    times[str(time._phase)] = float(time['_time-ms'])
            result.append(times)
        return result

    def getInventory(self, network, station='*', location='*', channel='*',
                     starttime=UTCDateTime(), endtime=UTCDateTime(),
                     instruments=True, min_latitude=-90, max_latitude=90,
                     min_longitude=-180, max_longitude=180,
                     modified_after=None, format='SUDS'):
        """
        Returns information about the available networks and stations in that
        particular space/time region.

        :type network: str
        :param network: Network code, e.g. ``'BW'``.
        :type station: str
        :param station: Station code, e.g. ``'MANZ'``. Station code may contain
            wild cards.
        :type location: str
        :param location: Location code, e.g. ``'01'``. Location code may
            contain wild cards.
        :type channel: str
        :param channel: Channel code, e.g. ``'EHE'``. Channel code may contain
            wild cards.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start date and time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End date and time.
        :type instruments: bool, optional
        :param instruments: Include instrument data. Default is ``True``.
        :type min_latitude: float, optional
        :param min_latitude: Minimum latitude, defaults to ``-90.0``.
        :type max_latitude: float, optional
        :param max_latitude: Maximum latitude, defaults to ``90.0``.
        :type min_longitude: float, optional
        :param min_longitude: Minimum longitude, defaults to ``-180.0``.
        :type max_longitude: float, optional
        :param max_longitude: Maximum longitude, defaults to ``180.0``.
        :type modified_after: :class:`~obspy.core.utcdatetime.UTCDateTime`,
            optional
        :param modified_after: Returns only data modified after given date.
            Default is ``None``, returning all available data.
        :type format: str, optional
        :param format: Output format, either ``'XML'`` or ``'SUDS'``. Either
            returns a XML document or a parsed SUDS object. Defaults to
            ``'SUDS'``.
        :return: XML document or a parsed SUDS object containing inventory
            information.

        .. rubric:: Example

        >>> from obspy.neries import Client
        >>> from obspy import UTCDateTime
        >>> client = Client(user='test@obspy.org')
        >>> dt = UTCDateTime("2011-01-01T00:00:00")
        >>> result = client.getInventory('GE', 'SNAA', '', 'BHZ', dt, dt+10,
        ...                              instruments=True)
        >>> paz = result.ArclinkInventory.inventory.responsePAZ
        >>> print(paz.poles)  # doctest: +ELLIPSIS
        (-0.037004,0.037016) (-0.037004,-0.037016) (-251.33,0.0) ...
        """
        # enable logging if debug option is set
        if self.debug:
            import logging
            logging.basicConfig(level=logging.INFO)
            logging.getLogger('suds.client').setLevel(logging.DEBUG)
        # initialize client
        client = SudsClient(SEISMOLINK_WSDL,
                            retxml=(format == 'XML'))
        # set prefixes for easier debugging
        client.add_prefix('gml', 'http://www.opengis.net/gml')
        client.add_prefix('ogc', 'http://www.opengis.net/ogc')
        client.add_prefix('xlin', 'http://www.w3.org/1999/xlink')
        client.add_prefix('urn', 'urn:xml:seisml:orfeus:neries:org')
        # set cache of 5 days
        cache = client.options.cache
        cache.setduration(days=5)
        # create user token
        usertoken = client.factory.create('UserTokenType')
        usertoken.email = self.user
        usertoken.password = self.password
        usertoken.label = self.user_agent.replace(' ', '_')
        usertoken.locale = ""
        # create station filter
        stationid = client.factory.create('StationIdentifierType')
        stationid.NetworkCode = network
        stationid.StationCode = station
        stationid.ChannelCode = channel
        stationid.LocId = location
        stationid.TimeSpan.TimePeriod.beginPosition = \
            UTCDateTime(starttime).strftime("%Y-%m-%dT%H:%M:%S")
        stationid.TimeSpan.TimePeriod.endPosition = \
            UTCDateTime(endtime).strftime("%Y-%m-%dT%H:%M:%S")
        # create spatial filters
        spatialbounds = client.factory.create('SpatialBoundsType')
        spatialbounds.BoundingBox.PropertyName = "e gero"
        spatialbounds.BoundingBox.Envelope.lowerCorner = "%f %f" %\
            (min(min_latitude, max_latitude),
             min(min_longitude, max_longitude))
        spatialbounds.BoundingBox.Envelope.upperCorner = "%f %f" %\
            (max(min_latitude, max_latitude),
             max(min_longitude, max_longitude))
        # instruments attribute
        if instruments:
            client.options.plugins.append(
                _AttributePlugin({'Instruments': 'true'}))
        else:
            client.options.plugins.append(
                _AttributePlugin({'Instruments': 'false'}))
        # modified_after attribute
        if modified_after:
            dt = UTCDateTime(modified_after).strftime("%Y-%m-%dT%H:%M:%S")
            client.options.plugins.append(
                _AttributePlugin({'ModifiedAfter': dt}))
        # add version attribute needed for instruments
        client.options.plugins.append(
            _AttributePlugin({'Version': '1.0'}))
        # request data
        response = client.service.getInventory(usertoken, stationid,
                                               spatialbounds)
        if format == 'XML':
            # response is a full SOAP response
            from xml.etree.ElementTree import fromstring, tostring
            temp = fromstring(response)
            xpath = '*/*/{urn:xml:seisml:orfeus:neries:org}ArclinkInventory'
            inventory = temp.find(xpath)
            # export XML prepending a XML declaration
            XML_DECLARATION = b"<?xml version='1.0' encoding='UTF-8'?>\n\n"
            return XML_DECLARATION + tostring(inventory, encoding='utf-8')
        else:
            # response is a SUDS object
            return response

    def getWaveform(self, network, station, location, channel, starttime,
                    endtime, format="MSEED"):
        """
        Retrieves waveform data from the NERIES Web service and returns a ObsPy
        Stream object.

        :type network: str
        :param network: Network code, e.g. ``'BW'``.
        :type station: str
        :param station: Station code, e.g. ``'MANZ'``.
        :type location: str
        :param location: Location code, e.g. ``'01'``. Location code may
            contain wild cards.
        :type channel: str
        :param channel: Channel code, e.g. ``'EHE'``. Channel code may contain
            wild cards.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start date and time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End date and time.
        :type format: str, optional
        :param format: Output format. Either as full SEED (``'FSEED'``) or
            Mini-SEED (``'MSEED'``) volume. Defaults to ``'MSEED'``.
        :return: ObsPy :class:`~obspy.core.stream.Stream` object.

        .. rubric:: Example

        >>> from obspy.neries import Client
        >>> client = Client(user='test@obspy.org')
        >>> dt = UTCDateTime("2009-04-01T00:00:00")
        >>> st = client.getWaveform("NL", "WIT", "", "BH*", dt, dt+30)
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        NL.WIT..BHZ | 2009-04-01T00:00:00.010200Z - ... | 40.0 Hz, 1201 samples
        NL.WIT..BHN | 2009-04-01T00:00:00.010200Z - ... | 40.0 Hz, 1201 samples
        NL.WIT..BHE | 2009-04-01T00:00:00.010200Z - ... | 40.0 Hz, 1201 samples
        """
        with NamedTemporaryFile() as tf:
            self.saveWaveform(tf._fileobj, network, station, location, channel,
                              starttime, endtime, format=format)
            # read stream using obspy.mseed
            tf.seek(0)
            try:
                stream = read(tf.name, 'MSEED')
            except:
                stream = Stream()
        # trim stream
        stream.trim(starttime, endtime)
        return stream

    def saveWaveform(self, filename, network, station, location, channel,
                     starttime, endtime, format="MSEED"):
        """
        Writes a retrieved waveform directly into a file.

        This method ensures the storage of the unmodified waveform data
        delivered by the NERIES Web service, e.g. preserving the record based
        quality flags of MiniSEED files which would be neglected reading it
        with :mod:`obspy.mseed`.

        :type filename: str
        :param filename: Name of the output file.
        :type network: str
        :param network: Network code, e.g. ``'BW'``.
        :type station: str
        :param station: Station code, e.g. ``'MANZ'``.
        :type location: str
        :param location: Location code, e.g. ``'01'``. Location code may
            contain wild cards.
        :type channel: str
        :param channel: Channel code, e.g. ``'EHE'``. Channel code may
            contain wild cards.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start date and time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End date and time.
        :type format: str, optional
        :param format: Output format. Either as full SEED (``'FSEED'``) or
            Mini-SEED (``'MSEED'``) volume. Defaults to ``'MSEED'``.
        :return: None

        .. seealso:: http://www.orfeus-eu.org/wsdl/seismolink/seismolink.wsdl

        .. rubric:: Example

        >>> from obspy.neries import Client
        >>> c = Client(user='test@obspy.org')
        >>> dt = UTCDateTime("2009-04-01T00:00:00")
        >>> st = c.saveWaveform("outfile.fseed", "NL", "WIT", "", "BH*",
        ...                     dt, dt+30, format="FSEED")  #doctest: +SKIP
        """
        # enable logging if debug option is set
        if self.debug:
            import logging
            logging.basicConfig(level=logging.INFO)
            logging.getLogger('suds.client').setLevel(logging.DEBUG)
        # initialize client
        client = SudsClient(SEISMOLINK_WSDL)
        # set cache of 5 days
        cache = client.options.cache
        cache.setduration(days=5)
        # create user token
        usertoken = client.factory.create('UserTokenType')
        usertoken.email = self.user
        usertoken.password = self.password
        usertoken.label = self.user_agent.replace(' ', '_')
        usertoken.locale = ""
        # create station filter
        stationid = client.factory.create('StationIdentifierType')
        stationid.NetworkCode = network
        stationid.StationCode = station
        stationid.ChannelCode = channel
        stationid.LocId = location
        # adding default record length (4096) * delta to start and end time to
        # ensure right date times
        # XXX: 4096 may be overkill
        delta = guessDelta(channel) * 4096
        stationid.TimeSpan.TimePeriod.beginPosition = \
            (UTCDateTime(starttime) - delta).strftime("%Y-%m-%dT%H:%M:%S")
        stationid.TimeSpan.TimePeriod.endPosition = \
            (UTCDateTime(endtime) + delta).strftime("%Y-%m-%dT%H:%M:%S")
        # request data
        if format == 'MSEED':
            client.options.plugins = \
                [_AttributePlugin({'DataFormat': 'MSEED'})]
        # start data request
        response = client.service.dataRequest(usertoken, stationid)
        client.options.plugins = []
        # filter for request ids
        request_ids = [r._Id for r in response.RoutedRequest]
        if not request_ids:
            return
        # check status using request ids
        _loops = 0
        while True:
            response = client.service.checkStatus(usertoken, request_ids)
            status = [r.ReadyFlag for r in response.RoutedRequest]
            # if we hit MAX_REQUESTS break the loop
            if _loops > MAX_REQUESTS:
                msg = 'MAX_REQUESTS exceeded - breaking current request loop'
                warnings.warn(msg, UserWarning)
                break
            if "false" in status:
                # retry until all are set to 'true'
                _loops += 1
                continue
            break
        # keep only request ids which are fulfilled and have 'status = OK'
        request_ids = [r._Id for r in response.RoutedRequest
                       if 'Status: OK' in r.StatusDescription and
                       r.Fulfillment == 100]
        if not request_ids:
            return
        # retrieve download URLs using request ids
        response = client.service.dataRetrieve(usertoken, request_ids)
        urls = [r.DownloadToken.DownloadURL for r in response.DataItem]
        # create file handler if a file name is given
        if isinstance(filename, (str, native_str)):
            fh = open(filename, "wb")
        elif hasattr(filename, "write"):
            fh = filename
        else:
            msg = "Parameter filename must be either string or file handler."
            raise TypeError(msg)
        for url in urls:
            fh.write(urllib.request.urlopen(url).read())
        if isinstance(filename, (str, native_str)):
            fh.close()
        # clean up
        response = client.service.purgeData(usertoken, request_ids)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
