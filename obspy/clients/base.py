# -*- coding: utf-8 -*-
"""
Base classes for uniform Client interfaces.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)


This module defines common interfaces for ObsPy client classes, using Abstract
Base Classes.  These common interfaces are a place to explicitly declare the
intent for any Client, regardless of its origin, and to return a Stream from
a get_waveforms method, a Catalog from a get_events method, and an Inventory
from a get_stations method. This encourages Client writers to connect their
data sources to Stream, Inventory, and Catalog types, and encourage users to
rely on them in their applications.  Four base classes are provided: one for
clients that return waveforms, one for those that return events, and one for
those that return stations.  Each inherits from a common base class, which
contains methods common to all.

Individual client classes inherit from one or more of WaveformClient,
EventClient, and StationClient, and re-program the get_waveforms, get_events,
and/or get_stations methods, like in the example below.


.. rubric:: Example

class MyNewClient(WaveformClient, StationClient):
    def __init__(self, url=None):
        self._version = '1.0'
        if url:
            self.conn = open(url)

    def get_service_version(self):
        self.conn.get_version()

    def get_waveforms(self, network, station, location, channel, starttime,
                      endtime):
        return self.conn.fetch_mseed(network, station, location, channel,
                                     starttime, endtime)

    def get_stations(self, network, station, location, channel, starttime,
                     endtime):
        return self.conn.fetch_inventory(network, station, location, channel,
                                         starttime, endtime)

"""
from abc import ABCMeta, abstractmethod
import io
import platform
import sys

import requests

import obspy


# Default user agents all HTTP clients should utilize.
encoding = sys.getdefaultencoding() or "UTF-8"
platform_ = platform.platform().encode(encoding).decode("ascii", "ignore")
# The default User Agent that will be sent with every request.
DEFAULT_USER_AGENT = "ObsPy/%s (%s, Python %s)" % (
    obspy.__version__, platform_, platform.python_version())
# The user agent tests should use by default.
DEFAULT_TESTING_USER_AGENT = "ObsPy/%s (test suite) (%s, Python %s)" % (
    obspy.__version__, platform_, platform.python_version())


class ClientException(Exception):
    """
    Base exception for Client classes.
    """
    pass


class ClientHTTPException(ClientException,
                          requests.exceptions.RequestException):
    """
    Exception that should be raised for all HTTP exceptions.

    Inherits from :class:`requests.exceptions.RequestException` so catching
    the main requests exception catches this one as well.
    """
    pass


class BaseClient(object):
    """
    Base class for common methods.
    """
    def __init__(self, debug=False):
        self._debug = debug


class RemoteBaseClient(BaseClient, metaclass=ABCMeta):
    def __init__(self, debug=False, timeout=120):
        """
        Base class for all remote mixin classes.

        :param debug: Passed on to the :class:`BaseClient` constructor.
        :type debug: bool
        :param timeout: The network timeout in seconds.
        :type timeout: float
        """
        self._timeout = timeout
        BaseClient.__init__(self, debug=debug)

    @abstractmethod
    def get_service_version(self):
        """
        Return a semantic version number of the remote service as a string.
        """
        pass


class HTTPClient(RemoteBaseClient, metaclass=ABCMeta):
    """
    Mix-in class to add HTTP capabilities.

    :param debug: Passed on to the :class:`BaseClient` constructor.
    :type debug: bool
    :param timeout: Passed on to the :class:`RemoteBaseClient` constructor.
    :type timeout: float

    .. rubric:: Example

    from obspy.clients.base import (WaveformClient, HTTPClient,
                                    DEFAULT_USER_AGENT)

    class NewClient(WaveformClient, HTTPClient):
        def __init__(self, user_agent=DEFAULT_USER_AGENT, debug=False,
                     timeout=20):
            HTTPClient.__init__(self, user_agent=user_agent, debug=debug,
                                timeout=timeout)

        def _handle_requests_http_error(self, r):
            r.raise_for_status()

        def get_service_version(self):
            ...

        def get_waveforms(...):
            ...
    """
    def __init__(self, debug=False, timeout=120,
                 user_agent=DEFAULT_USER_AGENT):
        self._user_agent = user_agent
        RemoteBaseClient.__init__(self, debug=debug, timeout=timeout)

    @abstractmethod
    def _handle_requests_http_error(self, r):
        """
        Error handling for the HTTP errors.

        Method called when the _download() method downloads something with a
        status code different than 200.

        The error codes mean different things for different web services
        thus this needs to be implemented by every HTTPClient.

        :param r: The response object resulting in the error.
        :type r: :class:`requests.Response`
        """
        pass

    def _download(self, url, params=None, filename=None, data=None,
                  content_type=None):
        """
        Download the URL with GET or POST and the chosen parameters.

        Will call the ``_handle_requests_http_error()`` method if the response
        comes back with an HTTP code other than 200. Returns the response
        object if successful and ``filename`` is not given - if given it will
        save the response to the specified file and return ``None``.

        By default it will send a GET request - if data is given it will
        send a POST request.

        :param url: The URL to download from.
        :type url: str
        :param params: Additional URL parameters.
        :type params: dict
        :param filename: String or file like object. Will download directly
            to the file. If specified, this function will return nothing.
        :type filename: str or file-like object
        :param data: If specified, a POST request will be sent with the data in
            the body of the request.
        :type data: dict, bytes, or file-like object
        :param content_type: Should only be relevant when ``data`` is specified
            and thus issuing a POST request. Can be used to set the
            ``Content-Type`` HTTP header to let the server know what type the
            body is, e.g. ``"text/plain"``.
        :type content_type: str
        :return: The response object assuming ``filename`` is ``None``.
        :rtype: :class:`requests.Response`
        """
        if params:
            params = {k: v for k, v in params.items()}

        _request_args = {"url": url,
                         "headers": {"User-Agent": self._user_agent},
                         "params": params,
                         "timeout": self._timeout}

        # Stream to file - no need to keep it in memory for large files.
        if filename:
            _request_args["stream"] = True

        if self._debug:
            # Construct the same URL requests would construct.
            from requests import PreparedRequest  # noqa
            p = PreparedRequest()
            # request doesnt use timeout parameter, it's used when actually
            # sending the request, but the request is never sent in this debug
            # block anyway, it's just for printing info on what would be sent
            p.prepare(
                method="GET",
                **{k: v for k, v in _request_args.items() if k != "timeout"})
            print("Downloading %s ..." % p.url)
            if data is not None:
                print("Sending along the following payload:")
                print("-" * 70)
                print(data.decode() if hasattr(data, "decode") else data)
                print("-" * 70)

        # Workaround for old request versions.
        try:
            if data is None:
                r = requests.get(**_request_args)
            else:
                # Compatibility with old request versions.
                if hasattr(data, "read"):
                    data = data.read()
                _request_args["data"] = data
                r = requests.post(**_request_args)
        except TypeError:
            if "stream" in _request_args:
                del _request_args["stream"]
            if data is None:
                r = requests.get(**_request_args)
            else:
                _request_args["data"] = data
                r = requests.post(**_request_args)

        # Only accept code 200.
        if r.status_code != 200:
            self._handle_requests_http_error(r)

        # Return if nothing else happens.
        if not filename:
            return r

        _chunk_size = 1024
        if hasattr(filename, "write"):
            for chunk in r.iter_content(chunk_size=_chunk_size):
                if not chunk:
                    continue
                filename.write(chunk)
        else:
            with io.open(filename, "wb") as fh:
                for chunk in r.iter_content(chunk_size=_chunk_size):
                    if not chunk:
                        continue
                    fh.write(chunk)


class WaveformClient(BaseClient, metaclass=ABCMeta):
    """
    Base class for Clients supporting Stream objects.
    """
    @abstractmethod
    def get_waveforms(self, *args, **kwargs):
        """
        Returns a Stream.

        Keyword arguments are passed to the underlying concrete class.

        :type network: str
        :param network: Select one or more network codes. Can be SEED network
            codes or data center defined codes. Multiple codes are
            comma-separated. Wildcards are allowed.
        :type station: str
        :param station: Select one or more SEED station codes. Multiple codes
            are comma-separated. Wildcards are allowed.
        :type location: str
        :param location: Select one or more SEED location identifiers. Multiple
            identifiers are comma-separated. Wildcards are allowed.
        :type channel: str
        :param channel: Select one or more SEED channel codes. Multiple codes
            are comma-separated.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Limit results to time series samples on or after the
            specified start time
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: Limit results to time series samples on or before the
            specified end time
        """
        pass


class EventClient(BaseClient, metaclass=ABCMeta):
    """
    Base class for Clients supporting Catalog objects.
    """
    @abstractmethod
    def get_events(self, *args, **kwargs):
        """
        Returns a Catalog.

        Keyword arguments are passed to the underlying concrete class.

        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param starttime: Limit to events on or after the specified start time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param endtime: Limit to events on or before the specified end time.
        :type minlatitude: float, optional
        :param minlatitude: Limit to events with a latitude larger than the
            specified minimum.
        :type maxlatitude: float, optional
        :param maxlatitude: Limit to events with a latitude smaller than the
            specified maximum.
        :type minlongitude: float, optional
        :param minlongitude: Limit to events with a longitude larger than the
            specified minimum.
        :type maxlongitude: float, optional
        :param maxlongitude: Limit to events with a longitude smaller than the
            specified maximum.
        :type latitude: float, optional
        :param latitude: Specify the latitude to be used for a radius search.
        :type longitude: float, optional
        :param longitude: Specify the longitude to the used for a radius
            search.
        :type minradius: float, optional
        :param minradius: Limit to events within the specified minimum number
            of degrees from the geographic point defined by the latitude and
            longitude parameters.
        :type maxradius: float, optional
        :param maxradius: Limit to events within the specified maximum number
            of degrees from the geographic point defined by the latitude and
            longitude parameters.
        :type mindepth: float, optional
        :param mindepth: Limit to events with depth more than the specified
            minimum.
        :type maxdepth: float, optional
        :param maxdepth: Limit to events with depth less than the specified
            maximum.
        :type minmagnitude: float, optional
        :param minmagnitude: Limit to events with a magnitude larger than the
            specified minimum.
        :type maxmagnitude: float, optional
        :param maxmagnitude: Limit to events with a magnitude smaller than the
            specified maximum.
        :type magnitudetype: str, optional
        :param magnitudetype: Specify a magnitude type to use for testing the
            minimum and maximum limits.
        """
        pass


class StationClient(BaseClient, metaclass=ABCMeta):
    """
    Base class for Clients supporting Inventory objects.
    """
    @abstractmethod
    def get_stations(self, *args, **kwargs):
        """
        Returns an Inventory.

        Keyword arguments are passed to the underlying concrete class.

        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Limit to metadata epochs starting on or after the
            specified start time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: Limit to metadata epochs ending on or before the
            specified end time.
        :type startbefore: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param startbefore: Limit to metadata epochs starting before specified
            time.
        :type startafter: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param startafter: Limit to metadata epochs starting after specified
            time.
        :type endbefore: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endbefore: Limit to metadata epochs ending before specified
            time.
        :type endafter: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endafter: Limit to metadata epochs ending after specified time.
        :type network: str
        :param network: Select one or more network codes. Can be SEED network
            codes or data center defined codes. Multiple codes are
            comma-separated.
        :type station: str
        :param station: Select one or more SEED station codes. Multiple codes
            are comma-separated.
        :type location: str
        :param location: Select one or more SEED location identifiers. Multiple
            identifiers are comma-separated. As a special case ``“--“`` (two
            dashes) will be translated to a string of two space characters to
            match blank location IDs.
        :type channel: str
        :param channel: Select one or more SEED channel codes. Multiple codes
            are comma-separated.
        :type minlatitude: float
        :param minlatitude: Limit to stations with a latitude larger than the
            specified minimum.
        :type maxlatitude: float
        :param maxlatitude: Limit to stations with a latitude smaller than the
            specified maximum.
        :type minlongitude: float
        :param minlongitude: Limit to stations with a longitude larger than the
            specified minimum.
        :type maxlongitude: float
        :param maxlongitude: Limit to stations with a longitude smaller than
            the specified maximum.
        :type latitude: float
        :param latitude: Specify the latitude to be used for a radius search.
        :type longitude: float
        :param longitude: Specify the longitude to the used for a radius
            search.
        :type minradius: float
        :param minradius: Limit results to stations within the specified
            minimum number of degrees from the geographic point defined by the
            latitude and longitude parameters.
        :type maxradius: float
        :param maxradius: Limit results to stations within the specified
            maximum number of degrees from the geographic point defined by the
            latitude and longitude parameters.
        """
        pass
