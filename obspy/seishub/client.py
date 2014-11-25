# -*- coding: utf-8 -*-
"""
SeisHub database client for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import PY2, native_str
from future import standard_library
with standard_library.hooks():
    import urllib.parse
    import urllib.request

from datetime import datetime
from lxml import objectify
from lxml.etree import Element, SubElement, tostring
from math import log
from obspy import UTCDateTime, Catalog, readEvents
from obspy.core.util import guessDelta
from obspy.xseed import Parser
import os
import pickle
import time
import warnings
import functools


HTTP_ACCEPTED_DATA_METHODS = ["PUT", "POST"]
HTTP_ACCEPTED_NODATA_METHODS = ["HEAD", "GET", "DELETE"]
HTTP_ACCEPTED_METHODS = HTTP_ACCEPTED_DATA_METHODS + \
    HTTP_ACCEPTED_NODATA_METHODS


KEYWORDS = {'network': 'network_id', 'station': 'station_id',
            'location': 'location_id', 'channel': 'channel_id',
            'starttime': 'start_datetime', 'endtime': 'end_datetime'}


def _unpickle(data):
    if PY2:
        obj = pickle.loads(data)
    else:
        # http://api.mongodb.org/python/current/\
        # python3.html#why-can-t-i-share-pickled-objectids-\
        # between-some-versions-of-python-2-and-3
        obj = pickle.loads(data, encoding="latin-1")
    return obj


def _callChangeGetPAZ(func):
    """
    This is a decorator to intercept a change in the arg list for
    seishub.client.station.getPAZ() with revision [3778].

    * throw a DeprecationWarning
    * make the correct call
    """
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        # function itself is first arg so len(args) == 3 means we got 2 args.
        if len(args) > 3:
            msg = "The arg/kwarg call syntax of getPAZ() has changed. " + \
                  "Please update your code! The old call syntax has been " + \
                  "deprecated and will stop working with the next version."
            warnings.warn(msg, DeprecationWarning)
            _self = args[0]
            network = args[1]
            station = args[2]
            datetime = args[3]
            args = args[4:]
            if len(args) == 0:
                location = kwargs.get('location', '')
                channel = kwargs.get('channel', '')
            elif len(args) == 1:
                location = args[0]
                channel = kwargs.get('channel', '')
            elif len(args) == 2:
                location = args[0]
                channel = args[1]
            if channel == "":
                msg = "Requesting PAZ for empty channel codes is not " + \
                      "supported anymore."
                warnings.warn(msg, UserWarning)
            seed_id = ".".join((network, station, location, channel))
            args = [_self, seed_id, datetime]
            kwargs = {}
        return func(*args, **kwargs)
    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func


class Client(object):
    """
    SeisHub database request Client class.

    The following classes are automatically linked with initialization.
    Follow the links in "Linked Class" for more information. They register
    via the name listed in "Entry Point".

    ===================  ====================================================
    Entry Point          Linked Class
    ===================  ====================================================
    ``Client.waveform``  :class:`~obspy.seishub.client._WaveformMapperClient`
    ``Client.station``   :class:`~obspy.seishub.client._StationMapperClient`
    ``Client.event``     :class:`~obspy.seishub.client._EventMapperClient`
    ===================  ====================================================

    .. rubric:: Example

    >>> from obspy.seishub import Client
    >>>
    >>> t = UTCDateTime("2009-09-03 00:00:00")
    >>> client = Client(timeout=2)
    >>>
    >>> st = client.waveform.getWaveform("BW", "RTBE", "", "EHZ", t, t + 20)
    >>> print(st)  # doctest: +ELLIPSIS
    1 Trace(s) in Stream:
    BW.RTBE..EHZ | 2009-09-03T00:00:00.000000Z - ... | 200.0 Hz, 4001 samples
    """
    def __init__(self, base_url="http://teide.geophysik.uni-muenchen.de:8080",
                 user="admin", password="admin", timeout=10, debug=False,
                 retries=3):
        """
        Initializes the SeisHub Web service client.

        :type base_url: str, optional
        :param base_url: SeisHub connection string. Defaults to
            'http://teide.geophysik.uni-muenchen.de:8080'.
        :type user: str, optional
        :param user: The user name used for identification with the Web
            service. Defaults to ``'admin'``.
        :type password: str, optional
        :param password: A password used for authentication with the Web
            service. Defaults to ``'admin'``.
        :type timeout: int, optional
        :param timeout: Seconds before a connection timeout is raised (default
            is 10 seconds). Available only for Python >= 2.6.x.
        :type debug: bool, optional
        :param debug: Enables verbose output.
        :type retries: int
        :param retries: Number of retries for failing requests.
        """
        self.base_url = base_url
        self.waveform = _WaveformMapperClient(self)
        self.station = _StationMapperClient(self)
        self.event = _EventMapperClient(self)
        self.timeout = timeout
        self.debug = debug
        self.retries = retries
        self.xml_seeds = {}
        self.station_list = {}
        # Create an OpenerDirector for Basic HTTP Authentication
        password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, base_url, user, password)
        auth_handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
        opener = urllib.request.build_opener(auth_handler)
        # install globally
        urllib.request.install_opener(opener)

    def ping(self):
        """
        Ping the SeisHub server.
        """
        try:
            t1 = time.time()
            urllib.request.urlopen(self.base_url, timeout=self.timeout).read()
            return (time.time() - t1) * 1000.0
        except:
            pass

    def testAuth(self):
        """
        Test if authentication information is valid. Raises an Exception if
        status code of response is not 200 (OK) or 401 (Forbidden).

        :rtype: bool
        :return: ``True`` if OK, ``False`` if invalid.
        """
        (code, _msg) = self._HTTP_request(self.base_url + "/xml/",
                                          method="HEAD")
        if code == 200:
            return True
        elif code == 401:
            return False
        else:
            raise Exception("Unexpected request status code: %s" % code)

    def _fetch(self, url, *args, **kwargs):  # @UnusedVariable
        params = {}
        # map keywords
        for key, value in KEYWORDS.items():
            if key in kwargs.keys():
                kwargs[value] = kwargs[key]
                del kwargs[key]
        # check for ranges and empty values
        for key, value in kwargs.items():
            if not value and value != 0:
                continue
            if isinstance(value, tuple) and len(value) == 2:
                params['min_' + str(key)] = str(value[0])
                params['max_' + str(key)] = str(value[1])
            elif isinstance(value, list) and len(value) == 2:
                params['min_' + str(key)] = str(value[0])
                params['max_' + str(key)] = str(value[1])
            else:
                params[str(key)] = str(value)
        # replace special characters
        remoteaddr = self.base_url + url + '?' + \
            urllib.parse.urlencode(params)
        if self.debug:
            print('\nRequesting %s' % (remoteaddr))
        # certain requests randomly fail on rare occasions, retry
        for _i in range(self.retries):
            try:
                response = urllib.request.urlopen(remoteaddr,
                                                  timeout=self.timeout)
                doc = response.read()
                return doc
            # XXX currently there are random problems with SeisHub's internal
            # XXX SQL database access ("cannot operate on a closed database").
            # XXX this can be circumvented by issuing the same request again..
            except Exception:
                continue
        response = urllib.request.urlopen(remoteaddr, timeout=self.timeout)
        doc = response.read()
        return doc

    def _HTTP_request(self, url, method, xml_string="", headers={}):
        """
        Send a HTTP request via urllib2.

        :type url: str
        :param url: Complete URL of resource
        :type method: str
        :param method: HTTP method of request, e.g. "PUT"
        :type headers: dict
        :param headers: Header information for request, e.g.
                {'User-Agent': "obspyck"}
        :type xml_string: str
        :param xml_string: XML for a send request (PUT/POST)
        """
        if method not in HTTP_ACCEPTED_METHODS:
            raise ValueError("Method must be one of %s" %
                             HTTP_ACCEPTED_METHODS)
        if method in HTTP_ACCEPTED_DATA_METHODS and not xml_string:
            raise TypeError("Missing data for %s request." % method)
        elif method in HTTP_ACCEPTED_NODATA_METHODS and xml_string:
            raise TypeError("Unexpected data for %s request." % method)

        req = _RequestWithMethod(method=method, url=url, data=xml_string,
                                 headers=headers)
        # it seems the following always ends in a HTTPError even with
        # nice status codes...?!?
        try:
            response = urllib.request.urlopen(req, timeout=self.timeout)
            return response.code, response.msg
        except urllib.request.HTTPError as e:
            return e.code, e.msg

    def _objectify(self, url, *args, **kwargs):
        doc = self._fetch(url, *args, **kwargs)
        return objectify.fromstring(doc)


class _BaseRESTClient(object):
    def __init__(self, client):
        self.client = client

    def getResource(self, resource_name, format=None, **kwargs):
        """
        Gets a resource.

        :type resource_name: str
        :param resource_name: Name of the resource.
        :type format: str, optional
        :param format: Format string, e.g. ``'xml'`` or ``'map'``.
        :return: Resource
        """
        # NOTHING goes ABOVE this line!
        for key, value in locals().items():
            if key not in ["self", "kwargs"]:
                kwargs[key] = value
        url = '/xml/' + self.package + '/' + self.resourcetype + '/' + \
              resource_name
        return self.client._fetch(url, **kwargs)

    def getXMLResource(self, resource_name, **kwargs):
        """
        Gets a XML resource.

        :type resource_name: str
        :param resource_name: Name of the resource.
        :return: Resource as :class:`lxml.objectify.ObjectifiedElement`
        """
        url = '/xml/' + self.package + '/' + self.resourcetype + '/' + \
              resource_name
        return self.client._objectify(url, **kwargs)

    def putResource(self, resource_name, xml_string, headers={}):
        """
        PUTs a XML resource.

        :type resource_name: str
        :param resource_name: Name of the resource.
        :type headers: dict
        :param headers: Header information for request,
            e.g. ``{'User-Agent': "obspyck"}``
        :type xml_string: str
        :param xml_string: XML for a send request (PUT/POST)
        :rtype: tuple
        :return: (HTTP status code, HTTP status message)

        .. rubric:: Example

        >>> c = Client()
        >>> xseed_file = "dataless.seed.BW_UH1.xml"
        >>> xml_str = open(xseed_file).read()  # doctest: +SKIP
        >>> c.station.putResource(xseed_file, xml_str)  # doctest: +SKIP
        (201, 'OK')
        """
        url = '/'.join([self.client.base_url, 'xml', self.package,
                        self.resourcetype, resource_name])
        return self.client._HTTP_request(
            url, method="PUT", xml_string=xml_string, headers=headers)

    def deleteResource(self, resource_name, headers={}):
        """
        DELETEs a XML resource.

        :type resource_name: str
        :param resource_name: Name of the resource.
        :type headers: dict
        :param headers: Header information for request,
            e.g. ``{'User-Agent': "obspyck"}``
        :return: (HTTP status code, HTTP status message)
        """
        url = '/'.join([self.client.base_url, 'xml', self.package,
                        self.resourcetype, resource_name])
        return self.client._HTTP_request(
            url, method="DELETE", headers=headers)


class _WaveformMapperClient(object):
    """
    Interface to access the SeisHub Waveform Web service.

    .. warning::
        This function should NOT be initialized directly, instead access the
        object via the :attr:`obspy.seishub.Client.waveform` attribute.

    .. seealso:: https://github.com/barsch/seishub.plugins.seismology/blob/\
master/seishub/plugins/seismology/waveform.py
    """
    def __init__(self, client):
        self.client = client

    def getNetworkIds(self, **kwargs):
        """
        Gets a list of network ids.

        :rtype: list
        :return: List of containing network ids.
        """
        url = '/seismology/waveform/getNetworkIds'
        root = self.client._objectify(url, **kwargs)
        return [str(node['network']) for node in root.getchildren()]

    def getStationIds(self, network=None, **kwargs):
        """
        Gets a list of station ids.

        :type network: str
        :param network: Network code, e.g. ``'BW'``.
        :rtype: list
        :return: List of containing station ids.
        """
        # NOTHING goes ABOVE this line!
        for key, value in locals().items():
            if key not in ["self", "kwargs"]:
                kwargs[key] = value
        url = '/seismology/waveform/getStationIds'
        root = self.client._objectify(url, **kwargs)
        return [str(node['station']) for node in root.getchildren()]

    def getLocationIds(self, network=None, station=None, **kwargs):
        """
        Gets a list of location ids.

        :type network: str
        :param network: Network code, e.g. ``'BW'``.
        :type station: str
        :param station: Station code, e.g. ``'MANZ'``.
        :rtype: list
        :return: List of containing location ids.
        """
        # NOTHING goes ABOVE this line!
        for key, value in locals().items():
            if key not in ["self", "kwargs"]:
                kwargs[key] = value
        url = '/seismology/waveform/getLocationIds'
        root = self.client._objectify(url, **kwargs)
        return [str(node['location']) for node in root.getchildren()]

    def getChannelIds(self, network=None, station=None, location=None,
                      **kwargs):
        """
        Gets a list of channel ids.

        :type network: str
        :param network: Network code, e.g. ``'BW'``.
        :type station: str
        :param station: Station code, e.g. ``'MANZ'``.
        :type location: str
        :param location: Location code, e.g. ``'00'``.
        :rtype: list
        :return: List of containing channel ids.
        """
        # NOTHING goes ABOVE this line!
        for key, value in locals().items():
            if key not in ["self", "kwargs"]:
                kwargs[key] = value
        url = '/seismology/waveform/getChannelIds'
        root = self.client._objectify(url, **kwargs)
        return [str(node['channel']) for node in root.getchildren()]

    def getLatency(self, network=None, station=None, location=None,
                   channel=None, **kwargs):
        """
        Gets a list of network latency values.

        :type network: str
        :param network: Network code, e.g. ``'BW'``.
        :type station: str
        :param station: Station code, e.g. ``'MANZ'``.
        :type location: str
        :param location: Location code, e.g. ``'00'``.
        :type channel: str
        :param channel: Channel code, e.g. ``'EHE'``.
        :rtype: list
        :return: List of dictionaries containing latency information.
        """
        # NOTHING goes ABOVE this line!
        for key, value in locals().items():
            if key not in ["self", "kwargs"]:
                kwargs[key] = value
        url = '/seismology/waveform/getLatency'
        root = self.client._objectify(url, **kwargs)
        return [dict(((k, v.pyval) for k, v in node.__dict__.items()))
                for node in root.getchildren()]

    def getWaveform(self, network, station, location=None, channel=None,
                    starttime=None, endtime=None, apply_filter=None,
                    getPAZ=False, getCoordinates=False,
                    metadata_timecheck=True, **kwargs):
        """
        Gets a ObsPy Stream object.

        :type network: str
        :param network: Network code, e.g. ``'BW'``.
        :type station: str
        :param station: Station code, e.g. ``'MANZ'``.
        :type location: str
        :param location: Location code, e.g. ``'00'``.
        :type channel: str
        :param channel: Channel code, supporting wildcard for component,
            e.g. ``'EHE'`` or ``'EH*'``.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start date and time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End date and time.
        :type apply_filter: bool, optional
        :param apply_filter: Apply filter (default is ``False``).
        :type getPAZ: bool, optional
        :param getPAZ: Fetch PAZ information and append to
            :class:`~obspy.core.trace.Stats` of all fetched traces. This
            considerably slows down the request (default is ``False``).
        :type getCoordinates: bool, optional
        :param getCoordinates: Fetch coordinate information and append to
            :class:`~obspy.core.trace.Stats` of all fetched traces. This
            considerably slows down the request (default is ``False``).
        :type metadata_timecheck: bool, optional
        :param metadata_timecheck: For ``getPAZ`` and ``getCoordinates`` check
            if metadata information is changing from start to end time. Raises
            an Exception if this is the case. This can be deactivated to save
            time.
        :rtype: :class:`~obspy.core.stream.Stream`
        :return: A ObsPy Stream object.
        """
        # NOTHING goes ABOVE this line!
        # append all args to kwargs, thus having everything in one dictionary
        for key, value in locals().items():
            if key not in ["self", "kwargs"]:
                kwargs[key] = value

        # allow time strings in arguments
        for time_ in ["starttime", "endtime"]:
            if isinstance(kwargs[time_], (str, native_str)):
                kwargs[time_] = UTCDateTime(kwargs[time_])

        trim_start = kwargs['starttime']
        trim_end = kwargs['endtime']
        # we expand the requested time span on both ends by two samples in
        # order to be able to make use of the nearest_sample option of
        # stream.trim(). (see trim() and tickets #95 and #105)
        # only possible if a channel is specified otherwise delta = 0
        delta = 2 * guessDelta(kwargs['channel'])
        kwargs['starttime'] = trim_start - delta
        kwargs['endtime'] = trim_end + delta

        url = '/seismology/waveform/getWaveform'
        data = self.client._fetch(url, **kwargs)
        if not data:
            raise Exception("No waveform data available")
        # unpickle
        stream = _unpickle(data)
        if len(stream) == 0:
            raise Exception("No waveform data available")
        stream._cleanup()

        # trimming needs to be done only if we extend the datetime above
        if channel:
            stream.trim(trim_start, trim_end)
        if getPAZ:
            for tr in stream:
                paz = self.client.station.getPAZ(seed_id=tr.id,
                                                 datetime=starttime)
                if metadata_timecheck:
                    paz_check = self.client.station.getPAZ(seed_id=tr.id,
                                                           datetime=endtime)
                    if paz != paz_check:
                        msg = "PAZ information changing from start time to" + \
                              " end time."
                        raise Exception(msg)
                tr.stats['paz'] = paz

        if getCoordinates:
            coords = self.client.station.getCoordinates(
                network=network, station=station, location=location,
                datetime=starttime)
            if metadata_timecheck:
                coords_check = self.client.station.getCoordinates(
                    network=network, station=station,
                    location=location, datetime=endtime)
                if coords != coords_check:
                    msg = "Coordinate information changing from start " + \
                          "time to end time."
                    raise Exception(msg)
            for tr in stream:
                tr.stats['coordinates'] = coords.copy()
        return stream

    def getPreview(self, network, station, location=None, channel=None,
                   starttime=None, endtime=None, trace_ids=None, **kwargs):
        """
        Gets a preview of a ObsPy Stream object.

        :type network: str
        :param network: Network code, e.g. ``'BW'``.
        :type station: str
        :param station: Station code, e.g. ``'MANZ'``.
        :type location: str
        :param location: Location code, e.g. ``'00'``.
        :type channel: str
        :param channel: Channel code, supporting wildcard for component,
            e.g. ``'EHE'`` or ``'EH*'``.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start date and time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End date and time.
        :rtype: :class:`~obspy.core.stream.Stream`
        :return: Waveform preview as ObsPy Stream object.
        """
        # NOTHING goes ABOVE this line!
        for key, value in locals().items():
            if key not in ["self", "kwargs"]:
                kwargs[key] = value

        url = '/seismology/waveform/getPreview'
        data = self.client._fetch(url, **kwargs)
        if not data:
            raise Exception("No waveform data available")
        # unpickle
        stream = _unpickle(data)
        return stream

    def getPreviewByIds(self, trace_ids=None, starttime=None, endtime=None,
                        **kwargs):
        """
        Gets a preview of a ObsPy Stream object.

        :type trace_ids: list
        :param trace_ids: List of trace IDs, e.g. ``['BW.MANZ..EHE']``.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start date and time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End date and time.
        :rtype: :class:`~obspy.core.stream.Stream`
        :return: Waveform preview as ObsPy Stream object.
        """
        # NOTHING goes ABOVE this line!
        for key, value in locals().items():
            if key not in ["self", "kwargs"]:
                kwargs[key] = value
        # concatenate list of IDs into string
        if 'trace_ids' in kwargs:
            if isinstance(kwargs['trace_ids'], list):
                kwargs['trace_ids'] = ','.join(kwargs['trace_ids'])
        url = '/seismology/waveform/getPreview'
        data = self.client._fetch(url, **kwargs)
        if not data:
            raise Exception("No waveform data available")
        # unpickle
        stream = _unpickle(data)
        return stream


class _StationMapperClient(_BaseRESTClient):
    """
    Interface to access the SeisHub Station Web service.

    .. warning::
        This function should NOT be initialized directly, instead access the
        object via the :attr:`obspy.seishub.Client.station` attribute.

    .. seealso:: https://github.com/barsch/seishub.plugins.seismology/blob/\
master/seishub/plugins/seismology/waveform.py
    """
    package = 'seismology'
    resourcetype = 'station'

    def getList(self, network=None, station=None, **kwargs):
        """
        Gets a list of station information.

        :type network: str
        :param network: Network code, e.g. ``'BW'``.
        :type station: str
        :param station: Station code, e.g. ``'MANZ'``.
        :rtype: list
        :return: List of dictionaries containing station information.
        """
        # NOTHING goes ABOVE this line!
        for key, value in locals().items():
            if key not in ["self", "kwargs"]:
                kwargs[key] = value
        url = '/seismology/station/getList'
        root = self.client._objectify(url, **kwargs)
        return [dict(((k, v.pyval) for k, v in node.__dict__.items()))
                for node in root.getchildren()]

    def getCoordinates(self, network, station, datetime, location=''):
        """
        Get coordinate information.

        Returns a dictionary with coordinate information for specified station
        at the specified time.

        :type network: str
        :param network: Network code, e.g. ``'BW'``.
        :type station: str
        :param station: Station code, e.g. ``'MANZ'``.
        :type datetime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param datetime: Time for which the PAZ is requested,
            e.g. ``'2010-01-01 12:00:00'``.
        :type location: str
        :param location: Location code, e.g. ``'00'``.
        :rtype: dict
        :return: Dictionary containing station coordinate information.
        """
        # NOTHING goes ABOVE this line!
        kwargs = {}  # no **kwargs so use empty dict
        for key, value in locals().items():
            if key not in ["self", "kwargs"]:
                kwargs[key] = value

        # try to read coordinates from previously obtained station lists
        netsta = ".".join([network, station])
        for data in self.client.station_list.get(netsta, []):
            # check if starttime is present and fitting
            if data['start_datetime'] == "":
                pass
            elif datetime < UTCDateTime(data['start_datetime']):
                continue
            # check if end time is present and fitting
            if data['end_datetime'] == "":
                pass
            elif datetime > UTCDateTime(data['end_datetime']):
                continue
            coords = {}
            for key in ['latitude', 'longitude', 'elevation']:
                coords[key] = data[key]
            return coords

        metadata = self.getList(**kwargs)
        if not metadata:
            msg = "No coordinates for station %s.%s at %s" % \
                (network, station, datetime)
            raise Exception(msg)
        stalist = self.client.station_list.setdefault(netsta, [])
        for data in metadata:
            if data not in stalist:
                stalist.append(data)
        if len(metadata) > 1:
            warnings.warn("Received more than one metadata set. Using first.")
        metadata = metadata[0]
        coords = {}
        for key in ['latitude', 'longitude', 'elevation']:
            coords[key] = metadata[key]
        return coords

    @_callChangeGetPAZ
    def getPAZ(self, seed_id, datetime):
        """
        Get PAZ for a station at given time span. Gain is the A0 normalization
        constant for the poles and zeros.

        :type seed_id: str
        :param seed_id: SEED or channel id, e.g. ``"BW.RJOB..EHZ"`` or
            ``"EHE"``.
        :type datetime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param datetime: Time for which the PAZ is requested,
            e.g. ``'2010-01-01 12:00:00'``.
        :rtype: dict
        :return: Dictionary containing zeros, poles, gain and sensitivity.

        .. rubric:: Example

        >>> c = Client(timeout=2)
        >>> paz = c.station.getPAZ('BW.MANZ..EHZ', '20090707')
        >>> paz['zeros']
        [0j, 0j]
        >>> len(paz['poles'])
        5
        >>> print(paz['poles'][0])
        (-0.037004+0.037016j)
        >>> paz['gain']
        60077000.0
        >>> paz['sensitivity']
        2516800000.0
        """
        # try to read PAZ from previously obtained XSEED data
        for res in self.client.xml_seeds.get(seed_id, []):
            parser = Parser(res)
            try:
                paz = parser.getPAZ(seed_id=seed_id,
                                    datetime=UTCDateTime(datetime))
                return paz
            except:
                continue
        network, station, location, channel = seed_id.split(".")
        # request station information
        station_list = self.getList(network=network, station=station,
                                    datetime=datetime)
        if not station_list:
            return {}
        # don't allow wild cards
        for wildcard in ['*', '?']:
            if wildcard in seed_id:
                msg = "Wildcards in seed_id are not allowed."
                raise ValueError(msg)

        if len(station_list) > 1:
            warnings.warn("Received more than one XSEED file. Using first.")

        xml_doc = station_list[0]
        res = self.client.station.getResource(xml_doc['resource_name'])
        reslist = self.client.xml_seeds.setdefault(seed_id, [])
        if res not in reslist:
            reslist.append(res)
        parser = Parser(res)
        paz = parser.getPAZ(seed_id=seed_id, datetime=UTCDateTime(datetime))
        return paz


class _EventMapperClient(_BaseRESTClient):
    """
    Interface to access the SeisHub Event Web service.

    .. warning::
        This function should NOT be initialized directly, instead access the
        object via the :attr:`obspy.seishub.Client.event` attribute.

    .. seealso:: https://github.com/barsch/seishub.plugins.seismology/blob/\
master/seishub/plugins/seismology/event.py
    """
    package = 'seismology'
    resourcetype = 'event'

    def getList(self, limit=50, offset=None, localisation_method=None,
                author=None, min_datetime=None, max_datetime=None,
                first_pick=None, last_pick=None, min_latitude=None,
                max_latitude=None, min_longitude=None, max_longitude=None,
                min_magnitude=None, max_magnitude=None, min_depth=None,
                max_depth=None, used_p=None, min_used_p=None, max_used_p=None,
                used_s=None, min_used_s=None, max_used_s=None,
                document_id=None, **kwargs):
        """
        Gets a list of event information.

        ..note:
            For SeisHub versions < 1.4 available keys include "user" and
            "account". In newer SeisHub versions they are replaced by "author".

        :rtype: list
        :return: List of dictionaries containing event information.

        The number of resulting events is by default limited to 50 entries from
        a SeisHub server. You may raise this by setting the ``limit`` option to
        a maximal value of 2500. Numbers above 2500 will result into an
        exception.
        """
        # check limit
        if limit > 2500:
            msg = "Maximal allowed limit is 2500 entries."
            raise ValueError(msg)
        # NOTHING goes ABOVE this line!
        for key, value in locals().items():
            if key not in ["self", "kwargs"]:
                kwargs[key] = value
        url = '/seismology/event/getList'
        root = self.client._objectify(url, **kwargs)
        results = [dict(((k, v.pyval) for k, v in node.__dict__.items()))
                   for node in root.getchildren()]
        for res in results:
            res['resource_name'] = str(res['resource_name'])
        if limit == len(results) or \
           limit is None and len(results) == 50 or \
           len(results) == 2500:
            msg = "List of results might be incomplete due to option 'limit'."
            warnings.warn(msg)
        return results

    def getEvents(self, **kwargs):
        """
        Fetches a catalog with event information. Parameters to narrow down
        the request are the same as for :meth:`getList`.

        ..warning::
            Only works when connecting to a SeisHub server of version 1.4.0
            or higher (serving event data as QuakeML).

        :rtype: :class:`~obspy.core.event.Catalog`
        :returns: Catalog containing event information matching the request.

        The number of resulting events is by default limited to 50 entries from
        a SeisHub server. You may raise this by setting the ``limit`` option to
        a maximal value of 2500. Numbers above 2500 will result into an
        exception.
        """
        resource_names = [item["resource_name"]
                          for item in self.getList(**kwargs)]
        cat = Catalog()
        for resource_name in resource_names:
            cat.extend(readEvents(self.getResource(resource_name)))
        return cat

    def getKML(self, nolabels=False, **kwargs):
        """
        Posts an event.getList() and returns the results as a KML file. For
        optional arguments, see documentation of
        :meth:`~obspy.seishub.client._EventMapperClient.getList()`

        :type nolabels: bool
        :param nolabels: Hide labels of events in KML. Can be useful with large
            data sets.
        :rtype: str
        :return: String containing KML information of all matching events. This
            string can be written to a file and loaded into e.g. Google Earth.
        """
        events = self.getList(**kwargs)
        timestamp = datetime.now()

        # construct the KML file
        kml = Element("kml")
        kml.set("xmlns", "http://www.opengis.net/kml/2.2")

        document = SubElement(kml, "Document")
        SubElement(document, "name").text = "SeisHub Event Locations"

        # style definitions for earthquakes
        style = SubElement(document, "Style")
        style.set("id", "earthquake")

        iconstyle = SubElement(style, "IconStyle")
        SubElement(iconstyle, "scale").text = "0.5"
        icon = SubElement(iconstyle, "Icon")
        SubElement(icon, "href").text = \
            "http://maps.google.com/mapfiles/kml/shapes/earthquake.png"
        hotspot = SubElement(iconstyle, "hotSpot")
        hotspot.set("x", "0.5")
        hotspot.set("y", "0")
        hotspot.set("xunits", "fraction")
        hotspot.set("yunits", "fraction")

        labelstyle = SubElement(style, "LabelStyle")
        SubElement(labelstyle, "color").text = "ff0000ff"
        SubElement(labelstyle, "scale").text = "0.8"

        folder = SubElement(document, "Folder")
        SubElement(folder, "name").text = "SeisHub Events (%s)" % \
                                          timestamp.date()
        SubElement(folder, "open").text = "1"

        # additional descriptions for the folder
        descrip_str = "Fetched from: %s" % self.client.base_url
        descrip_str += "\nFetched at: %s" % timestamp
        descrip_str += "\n\nSearch options:\n"
        descrip_str += "\n".join(["=".join((str(k), str(v)))
                                  for k, v in kwargs.items()])
        SubElement(folder, "description").text = descrip_str

        style = SubElement(folder, "Style")
        liststyle = SubElement(style, "ListStyle")
        SubElement(liststyle, "listItemType").text = "check"
        SubElement(liststyle, "bgColor").text = "00ffffff"
        SubElement(liststyle, "maxSnippetLines").text = "5"

        # add one marker per event
        interesting_keys = ['resource_name', 'localisation_method', 'account',
                            'user', 'public', 'datetime', 'longitude',
                            'latitude', 'depth', 'magnitude', 'used_p',
                            'used_s']
        for event_dict in events:
            placemark = SubElement(folder, "Placemark")
            date = str(event_dict['datetime']).split(" ")[0]
            mag = str(event_dict['magnitude'])

            # scale marker size to magnitude if this information is present
            if mag:
                mag = float(mag)
                label = "%s: %.1f" % (date, mag)
                try:
                    icon_size = 1.2 * log(1.5 + mag)
                except ValueError:
                    icon_size = 0.1
            else:
                label = date
                icon_size = 0.5
            if nolabels:
                SubElement(placemark, "name").text = ""
            else:
                SubElement(placemark, "name").text = label
            SubElement(placemark, "styleUrl").text = "#earthquake"
            style = SubElement(placemark, "Style")
            icon_style = SubElement(style, "IconStyle")
            liststyle = SubElement(style, "ListStyle")
            SubElement(liststyle, "maxSnippetLines").text = "5"
            SubElement(icon_style, "scale").text = str(icon_size)
            if event_dict['longitude'] and event_dict['latitude']:
                point = SubElement(placemark, "Point")
                SubElement(point, "coordinates").text = "%.10f,%.10f,0" % \
                    (event_dict['longitude'], event_dict['latitude'])

            # detailed information on the event for the description
            descrip_str = ""
            for key in interesting_keys:
                if key not in event_dict:
                    continue
                descrip_str += "\n%s: %s" % (key, event_dict[key])
            SubElement(placemark, "description").text = descrip_str

        # generate and return KML string
        return tostring(kml, pretty_print=True, xml_declaration=True)

    def saveKML(self, filename, overwrite=False, **kwargs):
        """
        Posts an event.getList() and writes the results as a KML file. For
        optional arguments, see help for
        :meth:`~obspy.seishub.client._EventMapperClient.getList()` and
        :meth:`~obspy.seishub.client._EventMapperClient.getKML()`

        :type filename: str
        :param filename: Filename (complete path) to save KML to.
        :type overwrite: bool
        :param overwrite: Overwrite existing file, otherwise if file exists an
            Exception is raised.
        :type nolabels: bool
        :param nolabels: Hide labels of events in KML. Can be useful with large
            data sets.
        :rtype: str
        :return: String containing KML information of all matching events. This
            string can be written to a file and loaded into e.g. Google Earth.
        """
        if not overwrite and os.path.lexists(filename):
            raise OSError("File %s exists and overwrite=False." % filename)
        kml_string = self.getKML(**kwargs)
        open(filename, "wt").write(kml_string)
        return


class _RequestWithMethod(urllib.request.Request):
    """
    Improved urllib2.Request Class for which the HTTP Method can be set to
    values other than only GET and POST.
    See http://benjamin.smedbergs.us/blog/2008-10-21/\
    putting-and-deleteing-in-python-urllib2/
    """
    def __init__(self, method, *args, **kwargs):
        if method not in HTTP_ACCEPTED_METHODS:
            msg = "HTTP Method not supported. " + \
                  "Supported are: %s." % HTTP_ACCEPTED_METHODS
            raise ValueError(msg)
        urllib.request.Request.__init__(self, *args, **kwargs)
        self._method = method

    def get_method(self):
        return self._method


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
