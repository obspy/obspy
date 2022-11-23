#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FDSN Web service client for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import collections.abc
import copy
import gzip
import io
import os
import re
from socket import timeout as socket_timeout
import textwrap
import threading
import warnings
from collections import OrderedDict
from http.client import HTTPException, IncompleteRead
from urllib.parse import urlparse

from lxml import etree

import obspy
from obspy import UTCDateTime, read_inventory
from .header import (DEFAULT_PARAMETERS, DEFAULT_USER_AGENT, FDSNWS,
                     OPTIONAL_PARAMETERS, PARAMETER_ALIASES,
                     URL_DEFAULT_SUBPATH, URL_MAPPINGS, URL_MAPPING_SUBPATHS,
                     WADL_PARAMETERS_NOT_TO_BE_PARSED, DEFAULT_SERVICES,
                     FDSNException, FDSNRedirectException, FDSNNoDataException,
                     FDSNTimeoutException,
                     FDSNNoAuthenticationServiceException,
                     FDSNBadRequestException, FDSNNoServiceException,
                     FDSNInternalServerException,
                     FDSNNotImplementedException,
                     FDSNBadGatewayException,
                     FDSNTooManyRequestsException,
                     FDSNRequestTooLargeException,
                     FDSNServiceUnavailableException,
                     FDSNUnauthorizedException,
                     FDSNForbiddenException,
                     FDSNDoubleAuthenticationException,
                     FDSNInvalidRequestException)
from .wadl_parser import WADLParser

from urllib.parse import urlencode
import urllib.request as urllib_request
import queue


DEFAULT_SERVICE_VERSIONS = {'dataselect': 1, 'station': 1, 'event': 1}


class CustomRedirectHandler(urllib_request.HTTPRedirectHandler):
    """
    Custom redirection handler to also do it for POST requests which the
    standard library does not do by default.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """
        Copied and modified from the standard library.
        """
        # Force the same behaviour for GET, HEAD, and POST.
        m = req.get_method()
        if (not (code in (301, 302, 303, 307) and
                 m in ("GET", "HEAD", "POST"))):
            raise urllib_request.HTTPError(req.full_url, code, msg, headers,
                                           fp)

        # be conciliant with URIs containing a space
        newurl = newurl.replace(' ', '%20')
        content_headers = ("content-length", "content-type")
        newheaders = dict((k, v) for k, v in req.headers.items()
                          if k.lower() not in content_headers)

        # Also redirect the data of the request which the standard library
        # interestingly enough does not do.
        return urllib_request.Request(
            newurl, headers=newheaders,
            data=req.data,
            origin_req_host=req.origin_req_host,
            unverifiable=True)


class NoRedirectionHandler(urllib_request.HTTPRedirectHandler):
    """
    Handler that does not direct!
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """
        Copied and modified from the standard library.
        """
        raise FDSNRedirectException(
            "Requests with credentials (username, password) are not being "
            "redirected by default to improve security. To force redirects "
            "and if you trust the data center, set `force_redirect` to True "
            "when initializing the Client.")


class Client(object):
    """
    FDSN Web service request client.

    For details see the :meth:`~obspy.clients.fdsn.client.Client.__init__()`
    method.
    """
    # Dictionary caching any discovered service. Therefore repeatedly
    # initializing a client with the same base URL is cheap.
    __service_discovery_cache = {}
    #: Regex for UINT8
    RE_UINT8 = r'(?:25[0-5]|2[0-4]\d|[0-1]?\d{1,2})'
    #: Regex for HEX4
    RE_HEX4 = r'(?:[\d,a-f]{4}|[1-9,a-f][0-9,a-f]{0,2}|0)'
    #: Regex for IPv4
    RE_IPv4 = r'(?:' + RE_UINT8 + r'(?:\.' + RE_UINT8 + r'){3})'
    #: Regex for IPv6
    RE_IPv6 = \
        r'(?:\[' + RE_HEX4 + r'(?::' + RE_HEX4 + r'){7}\]' + \
        r'|\[(?:' + RE_HEX4 + r':){0,5}' + RE_HEX4 + r'::\]' + \
        r'|\[::' + RE_HEX4 + r'(?::' + RE_HEX4 + r'){0,5}\]' + \
        r'|\[::' + RE_HEX4 + r'(?::' + RE_HEX4 + r'){0,3}:' + RE_IPv4 + \
        r'\]' + \
        r'|\[' + RE_HEX4 + r':' + \
        r'(?:' + RE_HEX4 + r':|:' + RE_HEX4 + r'){0,4}' + \
        r':' + RE_HEX4 + r'\])'
    #: Regex for checking the validity of URLs
    URL_REGEX = r'https?://' + \
                r'(' + RE_IPv4 + \
                r'|' + RE_IPv6 + \
                r'|localhost' + \
                r'|\w(?:[\w-]*\w)?' + \
                r'|(?:\w(?:[\w-]{0,61}[\w])?\.){1,}([a-z][a-z0-9-]{1,62}))' + \
                r'(?::\d{2,5})?' + \
                r'(/[\w\.-]+)*/?$'

    @classmethod
    def _validate_base_url(cls, base_url):
        if re.match(cls.URL_REGEX, base_url, re.IGNORECASE):
            return True
        else:
            return False

    def __init__(self, base_url="IRIS", major_versions=None, user=None,
                 password=None, user_agent=DEFAULT_USER_AGENT, debug=False,
                 timeout=120, service_mappings=None, force_redirect=False,
                 eida_token=None, _discover_services=True):
        """
        Initializes an FDSN Web Service client.

        >>> client = Client("IRIS")
        >>> print(client)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        FDSN Webservice Client (base url: http://service.iris.edu)
        Available Services: 'dataselect' (v...), 'event' (v...),
        'station' (v...), 'available_event_catalogs',
        'available_event_contributors'
        Use e.g. client.help('dataselect') for the
        parameter description of the individual services
        or client.help() for parameter description of
        all webservices.

        :type base_url: str
        :param base_url: Base URL of FDSN web service compatible server
            (e.g. "http://service.iris.edu") or key string for recognized
            server (one of %s).
        :type major_versions: dict
        :param major_versions: Allows to specify custom major version numbers
            for individual services (e.g.
            `major_versions={'station': 2, 'dataselect': 3}`), otherwise the
            latest version at time of implementation will be used.
        :type user: str
        :param user: User name of HTTP Digest Authentication for access to
            restricted data.
        :type password: str
        :param password: Password of HTTP Digest Authentication for access to
            restricted data.
        :type user_agent: str
        :param user_agent: The user agent for all requests.
        :type debug: bool
        :param debug: Debug flag.
        :type timeout: float
        :param timeout: Maximum time (in seconds) to wait for a single request
            to receive the first byte of the response (after which an exception
            is raised).
        :type service_mappings: dict
        :param service_mappings: For advanced use only. Allows the direct
            setting of the endpoints of the different services. (e.g.
            ``service_mappings={'station': 'http://example.com/test/stat/1'}``)
            Valid keys are ``event``, ``station``, and ``dataselect``. This
            will overwrite the ``base_url`` and ``major_versions`` arguments.
            For all services not specified, the default default locations
            indicated by ``base_url`` and ``major_versions`` will be used. Any
            service that is manually specified as ``None`` (e.g.
            ``service_mappings={'event': None}``) will be deactivated.
        :type force_redirect: bool
        :param force_redirect: By default the client will follow all HTTP
            redirects as long as no credentials (username and password)
            are given. If credentials are given it will raise an exception
            when a redirect is discovered. This is done to improve security.
            Settings this flag to ``True`` will force all redirects to be
            followed even if credentials are given.
        :type eida_token: str
        :param eida_token: Token for EIDA authentication mechanism, see
            http://geofon.gfz-potsdam.de/waveform/archive/auth/index.php. If a
            token is provided, options ``user`` and ``password`` must not be
            used. This mechanism is only available on select EIDA nodes. The
            token can be provided in form of the PGP message as a string, or
            the filename of a local file with the PGP message in it.
        :type _discover_services: bool
        :param _discover_services: By default the client will query information
            about the FDSN endpoint when it is instantiated.  In certain cases,
            this may place a heavy load on the FDSN service provider.  If set
            to ``False``, no service discovery is performed and default
            parameter support is assumed. This parameter is experimental and
            will likely be removed in the future.
        """
        self.debug = debug
        self.user = user
        self.timeout = timeout
        self._force_redirect = force_redirect

        # Cache for the webservice versions. This makes interactive use of
        # the client more convenient.
        self.__version_cache = {}

        if base_url.upper() in URL_MAPPINGS:
            url_mapping = base_url.upper()
            base_url = URL_MAPPINGS[url_mapping]
            url_subpath = URL_MAPPING_SUBPATHS.get(
                url_mapping, URL_DEFAULT_SUBPATH)
        else:
            if base_url.isalpha():
                msg = "The FDSN service shortcut `{}` is unknown."\
                      .format(base_url)
                raise ValueError(msg)
            url_subpath = URL_DEFAULT_SUBPATH
        # Make sure the base_url does not end with a slash.
        base_url = base_url.strip("/")
        # Catch invalid URLs to avoid confusing error messages
        if not self._validate_base_url(base_url):
            msg = "The FDSN service base URL `{}` is not a valid URL."\
                  .format(base_url)
            raise ValueError(msg)

        self.base_url = base_url
        self.url_subpath = url_subpath

        self._set_opener(user, password)

        self.request_headers = {"User-Agent": user_agent}
        # Avoid mutable kwarg.
        if major_versions is None:
            major_versions = {}
        # Make a copy to avoid overwriting the default service versions.
        self.major_versions = DEFAULT_SERVICE_VERSIONS.copy()
        self.major_versions.update(major_versions)

        # Avoid mutable kwarg.
        if service_mappings is None:
            service_mappings = {}
        self._service_mappings = service_mappings

        if self.debug is True:
            print("Base URL: %s" % self.base_url)
            if self._service_mappings:
                print("Custom service mappings:")
                for key, value in self._service_mappings.items():
                    print("\t%s: '%s'" % (key, value))
            print("Request Headers: %s" % str(self.request_headers))

        if _discover_services:
            self._discover_services()
        else:
            self.services = DEFAULT_SERVICES

        # Use EIDA token if provided - this requires setting new url openers.
        #
        # This can only happen after the services have been discovered as
        # the clients needs to know if the fdsnws implementation has support
        # for the EIDA token system.
        #
        # This is a non-standard feature but we support it, given the number
        # of EIDA nodes out there.
        if eida_token is not None:
            # Make sure user/pw are not also given.
            if user is not None or password is not None:
                msg = ("EIDA authentication token provided, but "
                       "user and password are also given.")
                raise FDSNDoubleAuthenticationException(msg)
            self.set_eida_token(eida_token)

    @property
    def _has_eida_auth(self):
        return self.services.get('eida-auth', False)

    def set_credentials(self, user, password):
        """
        Set user and password resulting in subsequent web service
        requests for waveforms being authenticated for potential access to
        restricted data.

        This will overwrite any previously set-up credentials/authentication.

        :type user: str
        :param user: User name of credentials.
        :type password: str
        :param password: Password for given user name.
        """
        self.user = user
        self._set_opener(user, password)

    def set_eida_token(self, token, validate=True):
        """
        Fetch user and password from the server using the provided token,
        resulting in subsequent web service requests for waveforms being
        authenticated for potential access to restricted data.
        This only works for select EIDA nodes and relies on the auth mechanism
        described here:
        http://geofon.gfz-potsdam.de/waveform/archive/auth/index.php

        This will overwrite any previously set-up credentials/authentication.

        :type token: str
        :param token: Token for EIDA authentication mechanism, see
            http://geofon.gfz-potsdam.de/waveform/archive/auth/index.php.
            This mechanism is only available on select EIDA nodes. The token
            can be provided in form of the PGP message as a string, or the
            filename of a local file with the PGP message in it.
        :type validate: bool
        :param validate: Whether to sanity check the token before sending it to
            the EIDA server or not.
        """
        user, password = self._resolve_eida_token(token, validate=validate)
        self.set_credentials(user, password)

    def _set_opener(self, user, password):
        # Only add the authentication handler if required.
        handlers = []
        if user is not None and password is not None:
            # Create an OpenerDirector for HTTP Digest Authentication
            password_mgr = urllib_request.HTTPPasswordMgrWithDefaultRealm()
            password_mgr.add_password(None, self.base_url, user, password)
            handlers.append(urllib_request.HTTPDigestAuthHandler(password_mgr))

        if (user is None and password is None) or self._force_redirect is True:
            # Redirect if no credentials are given or the force_redirect
            # flag is True.
            handlers.append(CustomRedirectHandler())
        else:
            handlers.append(NoRedirectionHandler())

        # Don't install globally to not mess with other codes.
        self._url_opener = urllib_request.build_opener(*handlers)
        if self.debug:
            print('Installed new opener with handlers: {!s}'.format(handlers))

    def _resolve_eida_token(self, token, validate=True):
        """
        Use the token to get credentials.
        """
        if not self._has_eida_auth:
            msg = ("EIDA token authentication requested but service at '{}' "
                   "does not specify /dataselect/auth in the "
                   "dataselect/application.wadl.").format(self.base_url)
            raise FDSNNoAuthenticationServiceException(msg)

        token_file = None
        # check if there's a local file that matches the provided string
        if os.path.isfile(token):
            token_file = token
            with open(token_file, 'rb') as fh:
                token = fh.read().decode()
        # sanity check on the token
        if validate:
            if not _validate_eida_token(token):
                if token_file:
                    msg = ("Read EIDA token from file '{}' but it does not "
                           "seem to contain a valid PGP message.").format(
                               token_file)
                else:
                    msg = ("EIDA token does not seem to be a valid PGP "
                           "message. If you passed a filename, make sure the "
                           "file actually exists.")
                raise ValueError(msg)

        # force https so that we don't send around tokens unsecurely
        url = 'https://{}{}/dataselect/1/auth'.format(
            urlparse(self.base_url).netloc + urlparse(self.base_url).path,
            self.url_subpath)
        # paranoid: check again that we only send the token to https
        if urlparse(url).scheme != "https":
            msg = 'This should not happen, please file a bug report.'
            raise Exception(msg)

        # Already does the error checking with fdsnws semantics.
        response = self._download(url=url, data=token.encode(),
                                  use_gzip=True, return_string=True,
                                  content_type='application/octet-stream')

        user, password = response.decode().split(':')
        if self.debug:
            print('Got temporary user/pw: {}/{}'.format(user, password))

        return user, password

    def get_events(self, starttime=None, endtime=None, minlatitude=None,
                   maxlatitude=None, minlongitude=None, maxlongitude=None,
                   latitude=None, longitude=None, minradius=None,
                   maxradius=None, mindepth=None, maxdepth=None,
                   minmagnitude=None, maxmagnitude=None, magnitudetype=None,
                   eventtype=None, includeallorigins=None,
                   includeallmagnitudes=None, includearrivals=None,
                   eventid=None, limit=None, offset=None, orderby=None,
                   catalog=None, contributor=None, updatedafter=None,
                   filename=None, **kwargs):
        """
        Query the event service of the client.

        >>> client = Client("IRIS")
        >>> cat = client.get_events(eventid=609301)
        >>> print(cat)
        1 Event(s) in Catalog:
        1997-10-14T09:53:11.070000Z | -22.145, -176.720 | 7.8 ...

        The return value is a :class:`~obspy.core.event.Catalog` object
        which can contain any number of events.

        >>> t1 = UTCDateTime("2001-01-07T00:00:00")
        >>> t2 = UTCDateTime("2001-01-07T03:00:00")
        >>> cat = client.get_events(starttime=t1, endtime=t2, minmagnitude=4,
        ...                         catalog="ISC")
        >>> print(cat)
        3 Event(s) in Catalog:
        2001-01-07T02:55:59.290000Z |  +9.801,  +76.548 | 4.9 ...
        2001-01-07T02:35:35.170000Z | -21.291,  -68.308 | 4.4 ...
        2001-01-07T00:09:25.630000Z | +22.946, -107.011 | 4.0 ...

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
        :param longitude: Specify the longitude to be used for a radius
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
        :param mindepth: Limit to events with depth, in kilometers, larger than
            the specified minimum.
        :type maxdepth: float, optional
        :param maxdepth: Limit to events with depth, in kilometers, smaller
            than the specified maximum.
        :type minmagnitude: float, optional
        :param minmagnitude: Limit to events with a magnitude larger than the
            specified minimum.
        :type maxmagnitude: float, optional
        :param maxmagnitude: Limit to events with a magnitude smaller than the
            specified maximum.
        :type magnitudetype: str, optional
        :param magnitudetype: Specify a magnitude type to use for testing the
            minimum and maximum limits.
        :type eventtype: str, optional
        :param eventtype: Limit to events with a specified event type.
            Multiple types are comma-separated (e.g.,
            ``"earthquake,quarry blast"``). Allowed values are from QuakeML.
            See :const:`obspy.core.event.header.EventType` for a list of
            allowed event types.
        :type includeallorigins: bool, optional
        :param includeallorigins: Specify if all origins for the event should
            be included, default is data center dependent but is suggested to
            be the preferred origin only.
        :type includeallmagnitudes: bool, optional
        :param includeallmagnitudes: Specify if all magnitudes for the event
            should be included, default is data center dependent but is
            suggested to be the preferred magnitude only.
        :type includearrivals: bool, optional
        :param includearrivals: Specify if phase arrivals should be included.
        :type eventid: str or int, optional
        :param eventid: Select a specific event by ID; event identifiers are
            data center specific (String or Integer).
        :type limit: int, optional
        :param limit: Limit the results to the specified number of events.
        :type offset: int, optional
        :param offset: Return results starting at the event count specified,
            starting at 1.
        :type orderby: str, optional
        :param orderby: Order the result by time or magnitude with the
            following possibilities:

            * time: order by origin descending time
            * time-asc: order by origin ascending time
            * magnitude: order by descending magnitude
            * magnitude-asc: order by ascending magnitude

        :type catalog: str, optional
        :param catalog: Limit to events from a specified catalog
        :type contributor: str, optional
        :param contributor: Limit to events contributed by a specified
            contributor.
        :type updatedafter: :class:`~obspy.core.utcdatetime.UTCDateTime`,
            optional
        :param updatedafter: Limit to events updated after the specified time.
        :type filename: str or file
        :param filename: If given, the downloaded data will be saved there
            instead of being parsed to an ObsPy object. Thus it will contain
            the raw data from the webservices.


        Any additional keyword arguments will be passed to the webservice as
        additional arguments. If you pass one of the default parameters and the
        webservice does not support it, a warning will be issued. Passing any
        non-default parameters that the webservice does not support will raise
        an error.
        """
        if "event" not in self.services:
            msg = "The current client does not have an event service."
            raise ValueError(msg)

        locs = locals()
        setup_query_dict('event', locs, kwargs)

        url = self._create_url_from_parameters(
            "event", DEFAULT_PARAMETERS['event'], kwargs)

        data_stream = self._download(url)
        data_stream.seek(0, 0)
        if filename:
            self._write_to_file_object(filename, data_stream)
            data_stream.close()
        else:
            cat = obspy.read_events(data_stream, format="quakeml")
            data_stream.close()
            return cat

    def get_stations(self, starttime=None, endtime=None, startbefore=None,
                     startafter=None, endbefore=None, endafter=None,
                     network=None, station=None, location=None, channel=None,
                     minlatitude=None, maxlatitude=None, minlongitude=None,
                     maxlongitude=None, latitude=None, longitude=None,
                     minradius=None, maxradius=None, level=None,
                     includerestricted=None, includeavailability=None,
                     updatedafter=None, matchtimeseries=None, filename=None,
                     format=None, **kwargs):
        """
        Query the station service of the FDSN client.

        >>> client = Client("IRIS")
        >>> starttime = UTCDateTime("2001-01-01")
        >>> endtime = UTCDateTime("2001-01-02")
        >>> inventory = client.get_stations(network="IU", station="A*",
        ...                                 starttime=starttime,
        ...                                 endtime=endtime)
        >>> print(inventory)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Inventory created at ...
            Created by: IRIS WEB SERVICE: fdsnws-station | version: ...
                        ...
            Sending institution: IRIS-DMC (IRIS-DMC)
            Contains:
                    Networks (1):
                            IU
                    Stations (3):
                            IU.ADK (Adak, Aleutian Islands, Alaska)
                            IU.AFI (Afiamalu, Samoa)
                            IU.ANMO (Albuquerque, New Mexico, USA)
                    Channels (0):
        >>> inventory.plot()  # doctest: +SKIP

        .. plot::

            from obspy import UTCDateTime
            from obspy.clients.fdsn import Client
            client = Client()
            starttime = UTCDateTime("2001-01-01")
            endtime = UTCDateTime("2001-01-02")
            inventory = client.get_stations(network="IU", station="A*",
                                            starttime=starttime,
                                            endtime=endtime)
            inventory.plot()


        The result is an :class:`~obspy.core.inventory.inventory.Inventory`
        object which models a StationXML file.

        The ``level`` argument determines the amount of returned information.
        ``level="station"`` is useful for availability queries whereas
        ``level="response"`` returns the full response information for the
        requested channels. ``level`` can furthermore be set to ``"network"``
        and ``"channel"``.

        >>> inventory = client.get_stations(
        ...     starttime=starttime, endtime=endtime,
        ...     network="IU", sta="ANMO", loc="00", channel="*Z",
        ...     level="response")
        >>> print(inventory)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Inventory created at ...
            Created by: IRIS WEB SERVICE: fdsnws-station | version: ...
                        ...
            Sending institution: IRIS-DMC (IRIS-DMC)
            Contains:
                Networks (1):
                    IU
                Stations (1):
                    IU.ANMO (Albuquerque, New Mexico, USA)
                Channels (4):
                    IU.ANMO.00.BHZ, IU.ANMO.00.LHZ, IU.ANMO.00.UHZ,
                    IU.ANMO.00.VHZ
        >>> inventory[0].plot_response(min_freq=1E-4)  # doctest: +SKIP

        .. plot::

            from obspy import UTCDateTime
            from obspy.clients.fdsn import Client
            client = Client()
            starttime = UTCDateTime("2001-01-01")
            endtime = UTCDateTime("2001-01-02")
            inventory = client.get_stations(
                starttime=starttime, endtime=endtime,
                network="IU", sta="ANMO", loc="00", channel="*Z",
                level="response")
            inventory[0].plot_response(min_freq=1E-4)

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
            comma-separated (e.g. ``"IU,TA"``).
        :type station: str
        :param station: Select one or more SEED station codes. Multiple codes
            are comma-separated (e.g. ``"ANMO,PFO"``).
        :type location: str
        :param location: Select one or more SEED location identifiers. Multiple
            identifiers are comma-separated (e.g. ``"00,01"``).  As a
            special case ``“--“`` (two dashes) will be translated to a string
            of two space characters to match blank location IDs.
        :type channel: str
        :param channel: Select one or more SEED channel codes. Multiple codes
            are comma-separated (e.g. ``"BHZ,HHZ"``).
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
        :param longitude: Specify the longitude to be used for a radius
            search.
        :type minradius: float
        :param minradius: Limit results to stations within the specified
            minimum number of degrees from the geographic point defined by the
            latitude and longitude parameters.
        :type maxradius: float
        :param maxradius: Limit results to stations within the specified
            maximum number of degrees from the geographic point defined by the
            latitude and longitude parameters.
        :type level: str
        :param level: Specify the level of detail for the results ("network",
            "station", "channel", "response"), e.g. specify "response" to get
            full information including instrument response for each channel.
        :type includerestricted: bool
        :param includerestricted: Specify if results should include information
            for restricted stations.
        :type includeavailability: bool
        :param includeavailability: Specify if results should include
            information about time series data availability.
        :type updatedafter: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param updatedafter: Limit to metadata updated after specified date;
            updates are data center specific.
        :type matchtimeseries: bool
        :param matchtimeseries: Only include data for which matching time
            series data is available.
        :type filename: str or file
        :param filename: If given, the downloaded data will be saved there
            instead of being parsed to an ObsPy object. Thus it will contain
            the raw data from the webservices.
        :type format: str
        :param format: The format in which to request station information.
            ``"xml"`` (StationXML) or ``"text"`` (FDSN station text format).
            XML has more information but text is much faster.

        :rtype: :class:`~obspy.core.inventory.inventory.Inventory`
        :returns: Inventory with requested station information.

        Any additional keyword arguments will be passed to the webservice as
        additional arguments. If you pass one of the default parameters and the
        webservice does not support it, a warning will be issued. Passing any
        non-default parameters that the webservice does not support will raise
        an error.
        """
        if "station" not in self.services:
            msg = "The current client does not have a station service."
            raise ValueError(msg)

        locs = locals()
        setup_query_dict('station', locs, kwargs)

        url = self._create_url_from_parameters(
            "station", DEFAULT_PARAMETERS['station'], kwargs)

        data_stream = self._download(url)
        data_stream.seek(0, 0)
        if filename:
            self._write_to_file_object(filename, data_stream)
            data_stream.close()
        else:
            # This works with XML and StationXML data.
            if format is None or format == 'xml':
                inventory = read_inventory(data_stream, format='STATIONXML')
            elif format == 'text':
                inventory = read_inventory(data_stream, format='STATIONTXT')
            else:
                inventory = read_inventory(data_stream)
            data_stream.close()
            return inventory

    def get_waveforms(self, network, station, location, channel, starttime,
                      endtime, quality=None, minimumlength=None,
                      longestonly=None, filename=None, attach_response=False,
                      **kwargs):
        """
        Query the dataselect service of the client.

        >>> client = Client("IRIS")
        >>> t1 = UTCDateTime("2010-02-27T06:30:00.000")
        >>> t2 = t1 + 5
        >>> st = client.get_waveforms("IU", "ANMO", "00", "LHZ", t1, t2)
        >>> print(st)  # doctest: +ELLIPSIS
        1 Trace(s) in Stream:
        IU.ANMO.00.LHZ | 2010-02-27T06:30:00.069538Z - ... | 1.0 Hz, 5 samples

        The services can deal with UNIX style wildcards.

        >>> st = client.get_waveforms("IU", "A*", "1?", "LHZ", t1, t2)
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        IU.ADK.10.LHZ  | 2010-02-27T06:30:00.069538Z - ... | 1.0 Hz, 5 samples
        IU.AFI.10.LHZ  | 2010-02-27T06:30:00.069538Z - ... | 1.0 Hz, 5 samples
        IU.ANMO.10.LHZ | 2010-02-27T06:30:00.069538Z - ... | 1.0 Hz, 5 samples

        Use ``attach_response=True`` to automatically add response information
        to each trace. This can be used to remove response using
        :meth:`~obspy.core.stream.Stream.remove_response`.

        >>> t = UTCDateTime("2012-12-14T10:36:01.6Z")
        >>> st = client.get_waveforms("TA", "E42A", "*", "BH?", t+300, t+400,
        ...                           attach_response=True)
        >>> st.remove_response(output="VEL") # doctest: +ELLIPSIS
        <obspy.core.stream.Stream object at ...>
        >>> st.plot()  # doctest: +SKIP

        .. plot::

            from obspy import UTCDateTime
            from obspy.clients.fdsn import Client
            client = Client("IRIS")
            t = UTCDateTime("2012-12-14T10:36:01.6Z")
            st = client.get_waveforms("TA", "E42A", "*", "BH?", t+300, t+400,
                                      attach_response=True)
            st.remove_response(output="VEL")
            st.plot()

        :type network: str
        :param network: Select one or more network codes. Can be SEED network
            codes or data center defined codes. Multiple codes are
            comma-separated (e.g. ``"IU,TA"``). Wildcards are allowed.
        :type station: str
        :param station: Select one or more SEED station codes. Multiple codes
            are comma-separated (e.g. ``"ANMO,PFO"``). Wildcards are allowed.
        :type location: str
        :param location: Select one or more SEED location identifiers. Multiple
            identifiers are comma-separated (e.g. ``"00,01"``). Wildcards are
            allowed.
        :type channel: str
        :param channel: Select one or more SEED channel codes. Multiple codes
            are comma-separated (e.g. ``"BHZ,HHZ"``).
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Limit results to time series samples on or after the
            specified start time
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: Limit results to time series samples on or before the
            specified end time
        :type quality: str, optional
        :param quality: Select a specific SEED quality indicator, handling is
            data center dependent.
        :type minimumlength: float, optional
        :param minimumlength: Limit results to continuous data segments of a
            minimum length specified in seconds.
        :type longestonly: bool, optional
        :param longestonly: Limit results to the longest continuous segment per
            channel.
        :type filename: str or file
        :param filename: If given, the downloaded data will be saved there
            instead of being parsed to an ObsPy object. Thus it will contain
            the raw data from the webservices.
        :type attach_response: bool
        :param attach_response: Specify whether the station web service should
            be used to automatically attach response information to each trace
            in the result set. A warning will be shown if a response can not be
            found for a channel. Does nothing if output to a file was
            specified.

        Any additional keyword arguments will be passed to the webservice as
        additional arguments. If you pass one of the default parameters and the
        webservice does not support it, a warning will be issued. Passing any
        non-default parameters that the webservice does not support will raise
        an error.
        """
        if "dataselect" not in self.services:
            msg = "The current client does not have a dataselect service."
            raise ValueError(msg)

        locs = locals()
        setup_query_dict('dataselect', locs, kwargs)

        # Special location handling. Convert empty strings to "--".
        if "location" in kwargs and not kwargs["location"]:
            kwargs["location"] = "--"

        url = self._create_url_from_parameters(
            "dataselect", DEFAULT_PARAMETERS['dataselect'], kwargs)

        # Gzip not worth it for MiniSEED and most likely disabled for this
        # route in any case.
        data_stream = self._download(url, use_gzip=False)
        data_stream.seek(0, 0)
        if filename:
            self._write_to_file_object(filename, data_stream)
            data_stream.close()
        else:
            st = obspy.read(data_stream, format="MSEED")
            data_stream.close()
            if attach_response:
                self._attach_responses(st)
            self._attach_dataselect_url_to_stream(st)
            st.trim(starttime, endtime)
            return st

    def _attach_responses(self, st):
        """
        Helper method to fetch response via get_stations() and attach it to
        each trace in stream.
        """
        netids = {}
        for tr in st:
            if tr.id not in netids:
                netids[tr.id] = (tr.stats.starttime, tr.stats.endtime)
                continue
            netids[tr.id] = (
                min(tr.stats.starttime, netids[tr.id][0]),
                max(tr.stats.endtime, netids[tr.id][1]))

        inventories = []
        for key, value in netids.items():
            net, sta, loc, chan = key.split(".")
            starttime, endtime = value
            try:
                inventories.append(self.get_stations(
                    network=net, station=sta, location=loc, channel=chan,
                    starttime=starttime, endtime=endtime, level="response"))
            except Exception as e:
                warnings.warn(str(e))
        st.attach_response(inventories)

    def get_waveforms_bulk(self, bulk, quality=None, minimumlength=None,
                           longestonly=None, filename=None,
                           attach_response=False, **kwargs):
        r"""
        Query the dataselect service of the client. Bulk request.

        Send a bulk request for waveforms to the server. `bulk` can either be
        specified as a filename, a file-like object or a string (with
        information formatted according to the FDSN standard) or a list of
        lists (each specifying network, station, location, channel, starttime
        and endtime). See examples and parameter description for more
        details.

        `bulk` can be provided in the following forms:

        (1) As a list of lists. Each list item has to be list of network,
            station, location, channel, starttime and endtime.

        (2) As a valid request string/file as defined in the
            `FDSNWS documentation <https://www.fdsn.org/webservices/>`_.
            The request information can be provided as a..

            - a string containing the request information
            - a string with the path to a local file with the request
            - an open file handle (or file-like object) with the request

        >>> client = Client("IRIS")
        >>> t1 = UTCDateTime("2010-02-27T06:30:00.000")
        >>> t2 = t1 + 1
        >>> t3 = t1 + 3
        >>> bulk = [("IU", "ANMO", "*", "BHZ", t1, t2),
        ...         ("IU", "AFI", "1?", "BHE", t1, t3),
        ...         ("GR", "GRA1", "*", "BH*", t2, t3)]
        >>> st = client.get_waveforms_bulk(bulk)
        >>> print(st)  # doctest: +ELLIPSIS
        5 Trace(s) in Stream:
        GR.GRA1..BHE   | 2010-02-27T06:30:01... | 20.0 Hz, 40 samples
        GR.GRA1..BHN   | 2010-02-27T06:30:01... | 20.0 Hz, 40 samples
        GR.GRA1..BHZ   | 2010-02-27T06:30:01... | 20.0 Hz, 40 samples
        IU.ANMO.00.BHZ | 2010-02-27T06:30:00... | 20.0 Hz, 20 samples
        IU.ANMO.10.BHZ | 2010-02-27T06:30:00... | 40.0 Hz, 40 samples
        >>> bulk = 'quality=B\n' + \
        ...        'longestonly=false\n' + \
        ...        'IU ANMO * BHZ 2010-02-27 2010-02-27T00:00:02\n' + \
        ...        'IU AFI 1? BHE 2010-02-27 2010-02-27T00:00:04\n' + \
        ...        'GR GRA1 * BH? 2010-02-27 2010-02-27T00:00:02\n'
        >>> st = client.get_waveforms_bulk(bulk)
        >>> print(st)  # doctest: +ELLIPSIS
        5 Trace(s) in Stream:
        GR.GRA1..BHE   | 2010-02-27T00:00:00... | 20.0 Hz, 40 samples
        GR.GRA1..BHN   | 2010-02-27T00:00:00... | 20.0 Hz, 40 samples
        GR.GRA1..BHZ   | 2010-02-27T00:00:00... | 20.0 Hz, 40 samples
        IU.ANMO.00.BHZ | 2010-02-27T00:00:00... | 20.0 Hz, 40 samples
        IU.ANMO.10.BHZ | 2010-02-27T00:00:00... | 40.0 Hz, 80 samples
        >>> st = client.get_waveforms_bulk("/tmp/request.txt") \
        ...     # doctest: +SKIP
        >>> print(st)  # doctest: +SKIP
        5 Trace(s) in Stream:
        GR.GRA1..BHE   | 2010-02-27T00:00:00... | 20.0 Hz, 40 samples
        GR.GRA1..BHN   | 2010-02-27T00:00:00... | 20.0 Hz, 40 samples
        GR.GRA1..BHZ   | 2010-02-27T00:00:00... | 20.0 Hz, 40 samples
        IU.ANMO.00.BHZ | 2010-02-27T00:00:00... | 20.0 Hz, 40 samples
        IU.ANMO.10.BHZ | 2010-02-27T00:00:00... | 40.0 Hz, 80 samples
        >>> t = UTCDateTime("2012-12-14T10:36:01.6Z")
        >>> t1 = t + 300
        >>> t2 = t + 400
        >>> bulk = [("TA", "S42A", "*", "BHZ", t1, t2),
        ...         ("TA", "W42A", "*", "BHZ", t1, t2),
        ...         ("TA", "Z42A", "*", "BHZ", t1, t2)]
        >>> st = client.get_waveforms_bulk(bulk, attach_response=True)
        >>> st.remove_response(output="VEL") # doctest: +ELLIPSIS
        <obspy.core.stream.Stream object at ...>
        >>> st.plot()  # doctest: +SKIP

        .. plot::

            from obspy import UTCDateTime
            from obspy.clients.fdsn import Client
            client = Client("IRIS")
            t = UTCDateTime("2012-12-14T10:36:01.6Z")
            t1 = t + 300
            t2 = t + 400
            bulk = [("TA", "S42A", "*", "BHZ", t1, t2),
                    ("TA", "W42A", "*", "BHZ", t1, t2),
                    ("TA", "Z42A", "*", "BHZ", t1, t2)]
            st = client.get_waveforms_bulk(bulk, attach_response=True)
            st.remove_response(output="VEL")
            st.plot()

        .. note::

            Use `attach_response=True` to automatically add response
            information to each trace. This can be used to remove response
            using :meth:`~obspy.core.stream.Stream.remove_response`.

        :type bulk: str, file or list[list]
        :param bulk: Information about the requested data. See above for
            details.
        :type quality: str, optional
        :param quality: Select a specific SEED quality indicator, handling is
            data center dependent. Ignored when `bulk` is provided as a
            request string/file.
        :type minimumlength: float, optional
        :param minimumlength: Limit results to continuous data segments of a
            minimum length specified in seconds. Ignored when `bulk` is
            provided as a request string/file.
        :type longestonly: bool, optional
        :param longestonly: Limit results to the longest continuous segment per
            channel. Ignored when `bulk` is provided as a request string/file.
        :type filename: str or file
        :param filename: If given, the downloaded data will be saved there
            instead of being parsed to an ObsPy object. Thus it will contain
            the raw data from the webservices.
        :type attach_response: bool
        :param attach_response: Specify whether the station web service should
            be used to automatically attach response information to each trace
            in the result set. A warning will be shown if a response can not be
            found for a channel. Does nothing if output to a file was
            specified.

        Any additional keyword arguments will be passed to the webservice as
        additional arguments. If you pass one of the default parameters and the
        webservice does not support it, a warning will be issued. Passing any
        non-default parameters that the webservice does not support will raise
        an error.
        """
        if "dataselect" not in self.services:
            msg = "The current client does not have a dataselect service."
            raise ValueError(msg)

        arguments = OrderedDict(
            quality=quality,
            minimumlength=minimumlength,
            longestonly=longestonly
        )
        bulk = get_bulk_string(bulk, arguments)

        url = self._build_url("dataselect", "query")

        data_stream = self._download(
            url, data=bulk, content_type='text/plain')
        data_stream.seek(0, 0)
        if filename:
            self._write_to_file_object(filename, data_stream)
            data_stream.close()
        else:
            st = obspy.read(data_stream, format="MSEED")
            data_stream.close()
            if attach_response:
                self._attach_responses(st)
            self._attach_dataselect_url_to_stream(st)
            return st

    def get_stations_bulk(self, bulk, level=None, includerestricted=None,
                          includeavailability=None, filename=None,
                          minlatitude=None, maxlatitude=None,
                          minlongitude=None, maxlongitude=None, latitude=None,
                          longitude=None, minradius=None, maxradius=None,
                          updatedafter=None, matchtimeseries=None, format=None,
                          **kwargs):
        """
        Query the station service of the client. Bulk request.

        Send a bulk request for stations to the server. `bulk` can either be
        specified as a filename, a file-like object or a string (with
        information formatted according to the FDSN standard) or a list of
        lists (each specifying network, station, location, channel, starttime
        and endtime). See examples and parameter description for more
        details.

        `bulk` can be provided in the following forms:

        (1) As a list of lists. Each list item has to be list of network,
            station, location, channel, starttime and endtime.

        (2) As a valid request string/file as defined in the
            `FDSNWS documentation <https://www.fdsn.org/webservices/>`_.
            The request information can be provided as a..

            - a string containing the request information
            - a string with the path to a local file with the request
            - an open file handle (or file-like object) with the request

        >>> client = Client("IRIS")
        >>> t1 = UTCDateTime("2010-02-27T06:30:00.000")
        >>> t2 = t1 + 1
        >>> t3 = t1 + 3
        >>> bulk = [("IU", "ANMO", "*", "BHZ", t1, t2),
        ...         ("IU", "AFI", "1?", "BHE", t1, t3),
        ...         ("GR", "GRA1", "*", "BH*", t2, t3)]
        >>> inv = client.get_stations_bulk(bulk)
        >>> print(inv)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Inventory created at ...
            Created by: IRIS WEB SERVICE: fdsnws-station | version: ...

            Sending institution: IRIS-DMC (IRIS-DMC)
            Contains:
                Networks (2):
                    GR, IU
                Stations (2):
                    GR.GRA1 (GRAFENBERG ARRAY, BAYERN)
                    IU.ANMO (Albuquerque, New Mexico, USA)
                Channels (0):

        >>> inv.plot()  # doctest: +SKIP

        .. plot::

            from obspy import UTCDateTime
            from obspy.clients.fdsn import Client

            client = Client("IRIS")
            t1 = UTCDateTime("2010-02-27T06:30:00.000")
            t2 = t1 + 1
            t3 = t1 + 3
            bulk = [("IU", "ANMO", "*", "BHZ", t1, t2),
                    ("IU", "AFI", "1?", "BHE", t1, t3),
                    ("GR", "GRA1", "*", "BH*", t2, t3)]
            inv = client.get_stations_bulk(bulk)
            inv.plot()

        >>> inv = client.get_stations_bulk(bulk, level="channel")
        >>> print(inv)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Inventory created at ...
            Created by: IRIS WEB SERVICE: fdsnws-station | version: ...

            Sending institution: IRIS-DMC (IRIS-DMC)
            Contains:
                Networks (2):
                    GR, IU
                Stations (2):
                    GR.GRA1 (GRAFENBERG ARRAY, BAYERN)
                    IU.ANMO (Albuquerque, New Mexico, USA)
                Channels (5):
                    GR.GRA1..BHZ, GR.GRA1..BHN, GR.GRA1..BHE, IU.ANMO.00.BHZ,
                    IU.ANMO.10.BHZ
        >>> inv = client.get_stations_bulk("/tmp/request.txt") \
        ...     # doctest: +SKIP
        >>> print(inv)  # doctest: +SKIP
        Inventory created at 2014-04-28T14:42:26.000000Z
            Created by: IRIS WEB SERVICE: fdsnws-station | version: 1.0.14
                    None
            Sending institution: IRIS-DMC (IRIS-DMC)
            Contains:
                Networks (2):
                    GR
                    IU
                Stations (2):
                    GR.GRA1 (GRAFENBERG ARRAY, BAYERN)
                    IU.ANMO (Albuquerque, New Mexico, USA)
                Channels (5):
                    GR.GRA1..BHE, GR.GRA1..BHN, GR.GRA1..BHZ, IU.ANMO.00.BHZ,
                    IU.ANMO.10.BHZ

        :type bulk: str, file or list[list]
        :param bulk: Information about the requested data. See above for
            details.
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
        :param longitude: Specify the longitude to be used for a radius
            search.
        :type minradius: float
        :param minradius: Limit results to stations within the specified
            minimum number of degrees from the geographic point defined by the
            latitude and longitude parameters.
        :type maxradius: float
        :param maxradius: Limit results to stations within the specified
            maximum number of degrees from the geographic point defined by the
            latitude and longitude parameters.
        :type level: str
        :param level: Specify the level of detail for the results ("network",
            "station", "channel", "response"), e.g. specify "response" to get
            full information including instrument response for each channel.
        :type includerestricted: bool
        :param includerestricted: Specify if results should include information
            for restricted stations.
        :type includeavailability: bool
        :param includeavailability: Specify if results should include
            information about time series data availability.
        :type updatedafter: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param updatedafter: Limit to metadata updated after specified date;
            updates are data center specific.
        :type matchtimeseries: bool
        :param matchtimeseries: Only include data for which matching time
            series data is available.
        :type filename: str or file
        :param filename: If given, the downloaded data will be saved there
            instead of being parsed to an ObsPy object. Thus it will contain
            the raw data from the webservices.
        :type format: str
        :param format: The format in which to request station information.
            ``"xml"`` (StationXML) or ``"text"`` (FDSN station text format).
            XML has more information but text is much faster.

        Any additional keyword arguments will be passed to the webservice as
        additional arguments. If you pass one of the default parameters and the
        webservice does not support it, a warning will be issued. Passing any
        non-default parameters that the webservice does not support will raise
        an error.
        """
        if "station" not in self.services:
            msg = "The current client does not have a station service."
            raise ValueError(msg)

        arguments = OrderedDict(
            minlatitude=minlatitude,
            maxlatitude=maxlatitude,
            minlongitude=minlongitude,
            maxlongitude=maxlongitude,
            latitude=latitude,
            longitude=longitude,
            minradius=minradius,
            maxradius=maxradius,
            level=level,
            includerestricted=includerestricted,
            includeavailability=includeavailability,
            updatedafter=updatedafter,
            matchtimeseries=matchtimeseries,
            format=format
        )
        bulk = get_bulk_string(bulk, arguments)

        url = self._build_url("station", "query")

        data_stream = self._download(
            url, data=bulk, content_type='text/plain')
        data_stream.seek(0, 0)
        if filename:
            self._write_to_file_object(filename, data_stream)
            data_stream.close()
            return
        else:
            # Works with text and StationXML data.
            if format is None or format == 'xml':
                inv = read_inventory(data_stream, format='STATIONXML')
            elif format == 'text':
                inv = read_inventory(data_stream, format='STATIONTXT')
            else:
                inv = read_inventory(data_stream)
            data_stream.close()
            return inv

    def _write_to_file_object(self, filename_or_object, data_stream):
        if hasattr(filename_or_object, "write"):
            filename_or_object.write(data_stream.read())
            return
        with open(filename_or_object, "wb") as fh:
            fh.write(data_stream.read())

    def _create_url_from_parameters(self, service, default_params, parameters):
        """
        """
        service_params = self.services[service]
        # Get all required parameters and make sure they are available!
        required_parameters = [
            key for key, value in service_params.items()
            if value["required"] is True]
        for req_param in required_parameters:
            if req_param not in parameters:
                msg = "Parameter '%s' is required." % req_param
                raise TypeError(msg)

        final_parameter_set = {}

        # Now loop over all parameters, convert them and make sure they are
        # accepted by the service.
        for key, value in parameters.items():
            if key not in service_params:
                # If it is not in the service but in the default parameters
                # raise a warning.
                if key in default_params:
                    msg = ("The standard parameter '%s' is not supported by "
                           "the webservice. It will be silently ignored." %
                           key)
                    warnings.warn(msg)
                    continue
                elif key in WADL_PARAMETERS_NOT_TO_BE_PARSED:
                    msg = ("The parameter '%s' is ignored because it is not "
                           "useful within ObsPy")
                    warnings.warn(msg % key)
                    continue
                # Otherwise raise an error.
                else:
                    msg = \
                        "The parameter '%s' is not supported by the service." \
                        % key
                    raise TypeError(msg)
            # Now attempt to convert the parameter to the correct type.
            this_type = service_params[key]["type"]

            # Try to decode to be able to work with bytes.
            if this_type is str:
                try:
                    value = value.decode()
                except AttributeError:
                    pass

            try:
                value = this_type(value)
            except Exception:
                msg = "'%s' could not be converted to type '%s'." % (
                    str(value), this_type.__name__)
                raise TypeError(msg)
            # Now convert to a string that is accepted by the webservice.
            value = convert_to_string(value)
            final_parameter_set[key] = value

        return self._build_url(service, "query",
                               parameters=final_parameter_set)

    def __str__(self):
        versions = dict([(s, self._get_webservice_versionstring(s))
                         for s in self.services if s in FDSNWS])
        services_string = ["'%s' (v%s)" % (s, versions[s])
                           for s in FDSNWS if s in self.services]
        other_services = sorted([s for s in self.services if s not in FDSNWS])
        services_string += ["'%s'" % s for s in other_services]
        services_string = ", ".join(services_string)
        ret = ("FDSN Webservice Client (base url: {url})\n"
               "Available Services: {services}\n\n"
               "Use e.g. client.help('dataselect') for the\n"
               "parameter description of the individual services\n"
               "or client.help() for parameter description of\n"
               "all webservices.".format(url=self.base_url,
                                         services=services_string))
        return ret

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def help(self, service=None):
        """
        Print a more extensive help for a given service.

        This will use the already parsed WADL files and be specific for each
        data center and always up-to-date.
        """
        if service is not None and service not in self.services:
            msg = "Service '%s' not available for current client." % service
            raise ValueError(msg)

        if service is None:
            services = list(self.services.keys())
        elif service in FDSNWS:
            services = [service]
        else:
            msg = "Service '%s is not a valid FDSN web service." % service
            raise ValueError(msg)

        msg = []
        for service in services:
            if service not in FDSNWS:
                continue
            service_default = DEFAULT_PARAMETERS[service]
            service_optional = OPTIONAL_PARAMETERS[service]

            msg.append("Parameter description for the "
                       "'%s' service (v%s) of '%s':" % (
                           service,
                           self._get_webservice_versionstring(service),
                           self.base_url))

            # Loop over all parameters and group them in four lists: available
            # default parameters, missing default parameters, optional
            # parameters and additional parameters.
            available_default_parameters = []
            missing_default_parameters = []
            optional_parameters = []
            additional_parameters = []

            printed_something = False

            for name in service_default:
                if name in self.services[service]:
                    available_default_parameters.append(name)
                else:
                    missing_default_parameters.append(name)

            for name in service_optional:
                if name in self.services[service]:
                    optional_parameters.append(name)

            defined_parameters = service_default + service_optional
            for name in self.services[service].keys():
                if name not in defined_parameters:
                    additional_parameters.append(name)

            def _param_info_string(name):
                param = self.services[service][name]
                name = "%s (%s)" % (name, param["type"].__name__.replace(
                    'new', ''))
                req_def = ""
                if param["required"]:
                    req_def = "Required Parameter"
                elif param["default_value"]:
                    req_def = "Default value: %s" % str(param["default_value"])
                if param["options"]:
                    req_def += ", Choices: %s" % \
                        ", ".join(map(str, param["options"]))
                if req_def:
                    req_def = ", %s" % req_def
                if param["doc_title"]:
                    doc_title = textwrap.fill(param["doc_title"], width=79,
                                              initial_indent="        ",
                                              subsequent_indent="        ",
                                              break_long_words=False)
                    doc_title = "\n" + doc_title
                else:
                    doc_title = ""

                return "    {name}{req_def}{doc_title}".format(
                    name=name, req_def=req_def, doc_title=doc_title)

            if optional_parameters:
                printed_something = True
                msg.append("The service offers the following optional "
                           "standard parameters:")
                for name in optional_parameters:
                    msg.append(_param_info_string(name))

            if additional_parameters:
                printed_something = True
                msg.append("The service offers the following "
                           "non-standard parameters:")
                for name in sorted(additional_parameters):
                    msg.append(_param_info_string(name))

            if missing_default_parameters:
                printed_something = True
                msg.append("WARNING: The service does not offer the following "
                           "standard parameters: %s" %
                           ", ".join(missing_default_parameters))

            if service == "event" and \
                    "available_event_catalogs" in self.services:
                printed_something = True
                msg.append("Available catalogs: %s" %
                           ", ".join(
                               self.services["available_event_catalogs"]))

            if service == "event" and \
                    "available_event_contributors" in self.services:
                printed_something = True
                msg.append("Available contributors: %s" %
                           ", ".join(
                               self.services["available_event_contributors"]))

            if printed_something is False:
                msg.append("No derivations from standard detected")

        print("\n".join(msg))

    def _download(self, url, return_string=False, data=None, use_gzip=True,
                  content_type=None):
        headers = self.request_headers.copy()
        if content_type:
            headers['Content-Type'] = content_type
        code, data = download_url(
            url, opener=self._url_opener, headers=headers,
            debug=self.debug, return_string=return_string, data=data,
            timeout=self.timeout, use_gzip=use_gzip)
        raise_on_error(code, data)
        return data

    def _build_url(self, service, resource_type, parameters={}):
        """
        Builds the correct URL.

        Replaces "query" with "queryauth" if client has authentication
        information.
        """
        # authenticated dataselect queries have different target URL
        if self.user is not None:
            if service == "dataselect" and resource_type == "query":
                resource_type = "queryauth"
        return build_url(self.base_url, service, self.major_versions[service],
                         resource_type, parameters,
                         service_mappings=self._service_mappings,
                         subpath=self.url_subpath)

    def _discover_services(self):
        """
        Automatically discovers available services.

        They are discovered by downloading the corresponding WADL files. If a
        WADL does not exist, the services are assumed to be non-existent.
        """
        services = ["dataselect", "event", "station"]
        # omit manually deactivated services
        for service, custom_target in self._service_mappings.items():
            if custom_target is None:
                services.remove(service)
        urls = [self._build_url(service, "application.wadl")
                for service in services]
        if "event" in services:
            urls.append(self._build_url("event", "catalogs"))
            urls.append(self._build_url("event", "contributors"))
        # Access cache if available.
        url_hash = frozenset(urls)
        if url_hash in self.__service_discovery_cache:
            if self.debug is True:
                print("Loading discovered services from cache.")
            self.services = copy.deepcopy(
                self.__service_discovery_cache[url_hash])
            return

        # Request all in parallel.
        wadl_queue = queue.Queue()

        headers = self.request_headers
        debug = self.debug
        opener = self._url_opener

        def get_download_thread(url):
            class ThreadURL(threading.Thread):
                def run(self):
                    # Catch 404s.
                    try:
                        code, data = download_url(
                            url, opener=opener, headers=headers,
                            debug=debug, timeout=self._timeout)
                        if code == 200:
                            wadl_queue.put((url, data))
                        # Pass on the redirect exception.
                        elif code is None and isinstance(
                                data, FDSNRedirectException):
                            wadl_queue.put((url, data))
                        else:
                            wadl_queue.put((url, None))
                    except urllib_request.HTTPError as e:
                        if e.code in [404, 502]:
                            wadl_queue.put((url, None))
                        else:
                            raise
                    except urllib_request.URLError:
                        wadl_queue.put((url, "timeout"))
                    except socket_timeout:
                        wadl_queue.put((url, "timeout"))
            threadurl = ThreadURL()
            threadurl._timeout = self.timeout
            return threadurl

        threads = list(map(get_download_thread, urls))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(15)
        self.services = {}

        # Collect the redirection exceptions to be able to raise nicer
        # exceptions.
        redirect_messages = set()

        for _ in range(wadl_queue.qsize()):
            item = wadl_queue.get()
            url, wadl = item

            # Just a safety measure.
            if hasattr(wadl, "decode"):
                decoded_wadl = wadl.decode('utf-8')
            else:
                decoded_wadl = wadl

            if wadl is None:
                continue
            elif isinstance(wadl, FDSNRedirectException):
                redirect_messages.add(str(wadl))
                continue
            elif decoded_wadl == "timeout":
                raise FDSNTimeoutException("Timeout while requesting '%s'."
                                           % url)

            if "dataselect" in url:
                wadl_parser = WADLParser(wadl)
                self.services["dataselect"] = wadl_parser.parameters
                # check if EIDA auth endpoint is in wadl
                # we need to attach it to the discovered services, as these are
                # later loaded from cache and just attaching an attribute to
                # this client won't help knowing later if EIDA auth is
                # supported at the server. a bit ugly but can't be helped.
                if wadl_parser._has_eida_auth:
                    self.services["eida-auth"] = True
                if self.debug is True:
                    print("Discovered dataselect service")
            elif "event" in url and "application.wadl" in url:
                self.services["event"] = WADLParser(wadl).parameters
                if self.debug is True:
                    print("Discovered event service")
            elif "station" in url:
                self.services["station"] = WADLParser(wadl).parameters
                if self.debug is True:
                    print("Discovered station service")
            elif "event" in url and "catalogs" in url:
                try:
                    self.services["available_event_catalogs"] = \
                        parse_simple_xml(wadl)["catalogs"]
                except ValueError:
                    msg = "Could not parse the catalogs at '%s'." % url
                    warnings.warn(msg)
            elif "event" in url and "contributors" in url:
                try:
                    self.services["available_event_contributors"] = \
                        parse_simple_xml(wadl)["contributors"]
                except ValueError:
                    msg = "Could not parse the contributors at '%s'." % url
                    warnings.warn(msg)
        if not self.services:
            if redirect_messages:
                raise FDSNRedirectException(", ".join(redirect_messages))

            msg = ("No FDSN services could be discovered at '%s'. This could "
                   "be due to a temporary service outage or an invalid FDSN "
                   "service address." % self.base_url)
            raise FDSNNoServiceException(msg)
        # Cache.
        if self.debug is True:
            print("Storing discovered services in cache.")
        self.__service_discovery_cache[url_hash] = \
            copy.deepcopy(self.services)

    def get_webservice_version(self, service):
        """
        Get full version information of webservice (as a tuple of ints).

        This method is cached and will only be called once for each service
        per client object.
        """
        if service is not None and service not in self.services:
            msg = "Service '%s' not available for current client." % service
            raise ValueError(msg)

        if service not in FDSNWS:
            msg = "Service '%s is not a valid FDSN web service." % service
            raise ValueError(msg)

        # Access cache.
        if service in self.__version_cache:
            return self.__version_cache[service]

        url = self._build_url(service, "version")
        version = self._download(url, return_string=True)
        version = list(map(int, version.split(b".")))

        # Store in cache.
        self.__version_cache[service] = version

        return version

    def _get_webservice_versionstring(self, service):
        """
        Get full version information of webservice as a string.
        """
        version = self.get_webservice_version(service)
        return ".".join(map(str, version))

    def _attach_dataselect_url_to_stream(self, st):
        """
        Attaches the actually used dataselet URL to each Trace.
        """
        url = self._build_url("dataselect", "query")
        for tr in st:
            tr.stats._fdsnws_dataselect_url = url


def convert_to_string(value):
    """
    Takes any value and converts it to a string compliant with the FDSN
    webservices.

    Will raise a ValueError if the value could not be converted.

    >>> print(convert_to_string("abcd"))
    abcd
    >>> print(convert_to_string(1))
    1
    >>> print(convert_to_string(1.2))
    1.2
    >>> print(convert_to_string( \
              UTCDateTime(2012, 1, 2, 3, 4, 5, 666666)))
    2012-01-02T03:04:05.666666
    >>> print(convert_to_string(True))
    true
    >>> print(convert_to_string(False))
    false
    """
    if isinstance(value, str):
        return value
    # Boolean test must come before integer check!
    elif isinstance(value, bool):
        return str(value).lower()
    elif isinstance(value, int):
        return str(value)
    elif isinstance(value, float):
        return str(value)
    elif isinstance(value, UTCDateTime):
        return str(value).replace("Z", "")
    else:
        raise TypeError("Unexpected type %s" % repr(value))


def build_url(base_url, service, major_version, resource_type,
              parameters=None, service_mappings=None, subpath='fdsnws'):
    """
    URL builder for the FDSN webservices.

    Built as a separate function to enhance testability.

    >>> print(build_url("http://service.iris.edu", "dataselect", 1, \
                        "application.wadl"))
    http://service.iris.edu/fdsnws/dataselect/1/application.wadl

    >>> print(build_url("http://service.iris.edu", "dataselect", 1, \
                        "query", {"cha": "EHE"}))
    http://service.iris.edu/fdsnws/dataselect/1/query?cha=EHE
    """
    # Avoid mutable kwargs.
    if parameters is None:
        parameters = {}
    if service_mappings is None:
        service_mappings = {}

    # Only allow certain resource types.
    if service not in ["dataselect", "event", "station"]:
        msg = "Resource type '%s' not allowed. Allowed types: \n%s" % \
            (service, ",".join(("dataselect", "event", "station")))
        raise ValueError(msg)

    # Special location handling.
    if "location" in parameters:
        loc = parameters["location"].replace(" ", "")
        # Empty location.
        if not loc:
            loc = "--"
        # Empty location at start of list.
        if loc.startswith(','):
            loc = "--" + loc
        # Empty location at end of list.
        if loc.endswith(','):
            loc += "--"
        # Empty location in middle of list.
        loc = loc.replace(",,", ",--,")
        parameters["location"] = loc

    # Apply per-service mappings if any.
    if service in service_mappings:
        url = "/".join((service_mappings[service], resource_type))
    else:
        if subpath is None:
            parts = (base_url, service, str(major_version),
                     resource_type)
        else:
            parts = (base_url, subpath.lstrip('/'), service,
                     str(major_version), resource_type)
        url = "/".join(parts)

    if parameters:
        # Strip parameters.
        for key, value in parameters.items():
            try:
                parameters[key] = value.strip()
            except Exception:
                pass
        url = "?".join((url, urlencode(parameters)))
    return url


def raise_on_error(code, data):
    """
    Raise an error for non-200 HTTP response codes

    :type code: int
    :param code: HTTP response code
    :type data: :class:`io.BytesIO`
    :param data: Data returned by the server
    """
    # get detailed server response message
    if code != 200:
        try:
            server_info = data.read()
        except Exception:
            server_info = None
        else:
            server_info = server_info.decode('ASCII', errors='ignore')
        if server_info:
            server_info = "\n".join(
                line for line in server_info.splitlines() if line)
    # No data.
    if code == 204:
        raise FDSNNoDataException("No data available for request.",
                                  server_info)
    elif code == 400:
        msg = ("Bad request. If you think your request was valid "
               "please contact the developers.")
        raise FDSNBadRequestException(msg, server_info)
    elif code == 401:
        raise FDSNUnauthorizedException("Unauthorized, authentication "
                                        "required.", server_info)
    elif code == 403:
        raise FDSNForbiddenException("Authentication failed.",
                                     server_info)
    elif code == 413:
        raise FDSNRequestTooLargeException("Request would result in too much "
                                           "data. Denied by the datacenter. "
                                           "Split the request in smaller "
                                           "parts", server_info)
    # Request URI too large.
    elif code == 414:
        msg = ("The request URI is too large. Please contact the ObsPy "
               "developers.", server_info)
        raise NotImplementedError(msg)
    elif code == 429:
        msg = ("Sent too many requests in a given amount of time ('rate "
               "limiting'). Wait before making a new request.")
        raise FDSNTooManyRequestsException(msg, server_info)
    elif code == 500:
        raise FDSNInternalServerException("Service responds: Internal server "
                                          "error", server_info)
    elif code == 501:
        raise FDSNNotImplementedException("Service responds: Not implemented ",
                                          server_info)
    elif code == 502:
        raise FDSNBadGatewayException("Service responds: Bad gateway ",
                                      server_info)
    elif code == 503:
        raise FDSNServiceUnavailableException("Service temporarily "
                                              "unavailable",
                                              server_info)
    elif code is None:
        if "timeout" in str(data).lower() or "timed out" in str(data).lower():
            raise FDSNTimeoutException("Timed Out")
        else:
            raise FDSNException("Unknown Error (%s): %s" % (
                (str(data.__class__.__name__), str(data))))
    # Catch any non 200 codes.
    elif code != 200:
        raise FDSNException("Unknown HTTP code: %i" % code, server_info)


def download_url(url, opener, timeout=10, headers={}, debug=False,
                 return_string=True, data=None, use_gzip=True):
    """
    Returns a pair of tuples.

    The first one is the returned HTTP code and the second the data as
    string.

    Will return a tuple of Nones if the service could not be found.
    All encountered exceptions will get raised unless `debug=True` is
    specified.

    Performs a http GET if data=None, otherwise a http POST.
    """
    if debug is True:
        print("Downloading %s %s requesting gzip compression" % (
            url, "with" if use_gzip else "without"))
        if data:
            print("Sending along the following payload:")
            print("-" * 70)
            print(data.decode())
            print("-" * 70)
    try:
        request = urllib_request.Request(url=url, headers=headers)
        # Request gzip encoding if desired.
        if use_gzip:
            request.add_header("Accept-encoding", "gzip")
        url_obj = opener.open(request, timeout=timeout, data=data)
    # Catch HTTP errors.
    except urllib_request.HTTPError as e:
        if debug is True:
            msg = "HTTP error %i, reason %s, while downloading '%s': %s" % \
                  (e.code, str(e.reason), url, e.read())
            print(msg)
        else:
            # Without this line we will get unclosed sockets
            e.read()
        return e.code, e
    except Exception as e:
        if debug is True:
            print("Error while downloading: %s" % url)
        return None, e

    code = url_obj.getcode()

    # Unpack gzip if necessary.
    if url_obj.info().get("Content-Encoding") == "gzip":
        if debug is True:
            print("Uncompressing gzipped response for %s" % url)
        # Cannot directly stream to gzip from urllib!
        # http://www.enricozini.org/2011/cazzeggio/python-gzip/
        try:
            reader = url_obj.read()
        except IncompleteRead:
            msg = 'Problem retrieving data from datacenter. '
            msg += 'Try reducing size of request.'
            raise HTTPException(msg)
        buf = io.BytesIO(reader)
        buf.seek(0, 0)
        f = gzip.GzipFile(fileobj=buf)
    else:
        f = url_obj

    if return_string is False:
        data = io.BytesIO(f.read())
    else:
        data = f.read()

    if debug is True:
        print("Downloaded %s with HTTP code: %i" % (url, code))

    return code, data


def setup_query_dict(service, locs, kwargs):
    """
    """
    # check if alias is used together with the normal parameter
    for key in kwargs:
        if key in PARAMETER_ALIASES:
            if locs[PARAMETER_ALIASES[key]] is not None:
                msg = ("two parameters were provided for the same option: "
                       "%s, %s" % (key, PARAMETER_ALIASES[key]))
                raise FDSNInvalidRequestException(msg)
    # short aliases are not mentioned in the downloaded WADLs, so we have
    # to map it here according to the official FDSN WS documentation
    for key in list(kwargs.keys()):
        if key in PARAMETER_ALIASES:
            value = kwargs.pop(key)
            if value is not None:
                kwargs[PARAMETER_ALIASES[key]] = value

    for param in DEFAULT_PARAMETERS[service] + OPTIONAL_PARAMETERS[service]:
        param = PARAMETER_ALIASES.get(param, param)
        value = locs[param]
        if value is not None:
            kwargs[param] = value


def parse_simple_xml(xml_string):
    """
    Simple helper function for parsing the Catalog and Contributor availability
    files.

    Parses XMLs of the form::

        <Bs>
            <total>4</total>
            <B>1</B>
            <B>2</B>
            <B>3</B>
            <B>4</B>
        </Bs>

    and return a dictionary with a single item::

        {"Bs": set(("1", "2", "3", "4"))}
    """
    root = etree.fromstring(xml_string.strip())

    if not root.tag.endswith("s"):
        msg = "Could not parse the XML."
        raise ValueError(msg)
    child_tag = root.tag[:-1]
    children = [i.text for i in root if i.tag == child_tag]

    return {root.tag.lower(): set(children)}


def get_bulk_string(bulk, arguments):
    if not bulk:
        msg = ("Empty 'bulk' parameter potentially leading to a FDSN request "
               "of all available data")
        raise FDSNInvalidRequestException(msg)
    # If its an iterable, we build up the query string from it
    # StringIO objects also have __iter__ so check for 'read' as well
    if isinstance(bulk, collections.abc.Iterable) \
            and not hasattr(bulk, "read") \
            and not isinstance(bulk, str):
        tmp = ["%s=%s" % (key, convert_to_string(value))
               for key, value in arguments.items() if value is not None]
        # empty location codes have to be represented by two dashes
        tmp += [" ".join((net, sta, loc or "--", cha,
                          convert_to_string(t1), convert_to_string(t2)))
                for net, sta, loc, cha, t1, t2 in bulk]
        bulk = "\n".join(tmp)
    else:
        if any([value is not None for value in arguments.values()]):
            msg = ("Parameters %s are ignored when request data is "
                   "provided as a string or file!")
            warnings.warn(msg % arguments.keys())
        # if it has a read method, read data from there
        if hasattr(bulk, "read"):
            bulk = bulk.read()
        elif isinstance(bulk, str):
            # check if bulk is a local file
            if "\n" not in bulk and os.path.isfile(bulk):
                with open(bulk, 'r') as fh:
                    tmp = fh.read()
                bulk = tmp
            # just use bulk as input data
            else:
                pass
        else:
            msg = ("Unrecognized input for 'bulk' argument. Please "
                   "contact developers if you think this is a bug.")
            raise NotImplementedError(msg)

    if hasattr(bulk, "encode"):
        bulk = bulk.encode("ascii")
    return bulk


def _validate_eida_token(token):
    """
    Just a basic check if the string contains something that looks like a PGP
    message
    """
    if re.search(pattern='BEGIN PGP MESSAGE', string=token,
                 flags=re.IGNORECASE):
        return True
    elif re.search(pattern='BEGIN PGP SIGNED MESSAGE', string=token,
                   flags=re.IGNORECASE):
        return True
    return False


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
