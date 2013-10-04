#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FDSN Web service client for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from io import BytesIO
from lxml import etree
import obspy
from obspy import UTCDateTime
from obspy.fdsn.wadl_parser import WADLParser
from obspy.fdsn.header import DEFAULT_USER_AGENT, \
    URL_MAPPINGS, DEFAULT_PARAMETERS, PARAMETER_ALIASES, \
    WADL_PARAMETERS_NOT_TO_BE_PARSED, FDSNException
from obspy.core.util.misc import wrap_long_string

import Queue
import threading
import urllib
import urllib2
import warnings


class Client(object):
    """
    FDSN Web service request client.

    >>> client = Client("IRIS")
    >>> print client
    FDSN Webservice Client (base url: http://service.iris.edu)
    Available Services: 'dataselect', 'station', 'event'
    <BLANKLINE>
    Use e.g. client.help('dataselect') for the
    parameter description of the individual services
    or client.help() for parameter description of
    all webservices.

    For details see :meth:`__init__`.
    """
    def __init__(self, base_url="IRIS", major_version=1, user=None,
                 password=None, user_agent=DEFAULT_USER_AGENT, debug=False):
        """
        Initializes an FDSN Web Service client.

        >>> client = Client("IRIS")
        >>> print client
        FDSN Webservice Client (base url: http://service.iris.edu)
        Available Services: 'dataselect', 'station', 'event'
        <BLANKLINE>
        Use e.g. client.help('dataselect') for the
        parameter description of the individual services
        or client.help() for parameter description of
        all webservices.

        :type base_url: str
        :param base_url: Base URL of FDSN web service compatible server
            (e.g. "http://service.iris.edu") or key string for recognized
            server (currently "IRIS", "USGS", "RESIF", "NCEDC")
        :type major_version: int
        :param major_version: Major version number of server to access.
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
        """
        self.debug = debug
        #if user and password:
            ## Create an OpenerDirector for HTTP Digest Authentication
            #password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
            #password_mgr.add_password(None, base_url, user, password)
            #auth_handler = urllib2.HTTPDigestAuthHandler(password_mgr)
            #opener = urllib2.build_opener(auth_handler)
            ## install globally
            #urllib2.install_opener(opener)

        if base_url.upper() in URL_MAPPINGS:
            self.base_url = URL_MAPPINGS[base_url.upper()]
        else:
            self.base_url = base_url
        # Make sure the base_url does not end with a slash.
        self.base_url = self.base_url.strip("/")

        self.request_headers = {"User-Agent": user_agent}
        self.major_version = major_version

        if self.debug is True:
            print "Base URL: %s" % self.base_url
            print "Request Headers: %s" % str(self.request_headers)

        self._discover_services()

    def get_events(self, starttime=None, endtime=None, minlatitude=None,
                   maxlatitude=None, minlongitude=None, maxlongitude=None,
                   latitude=None, longitude=None, minradius=None,
                   maxradius=None, mindepth=None, maxdepth=None,
                   minmagnitude=None, maxmagnitude=None, magnitudetype=None,
                   includeallorigins=None, includeallmagnitudes=None,
                   includearrivals=None, eventid=None, limit=None, offset=None,
                   orderby=None, catalog=None, contributor=None,
                   updatedafter=None, filename=None, **kwargs):
        """
        Query the event service of the client.

        >>> client = Client("IRIS")
        >>> cat = client.get_events(eventid=609301)
        >>> print cat
        1 Event(s) in Catalog:
        1997-10-14T09:53:11.070000Z | -22.145, -176.720 | 7.8 mw
        >>> t1 = UTCDateTime("2011-01-07T01:00:00")
        >>> t2 = UTCDateTime("2011-01-07T02:00:00")
        >>> cat = client.get_events(starttime=t1, endtime=t2, minmagnitude=4)
        >>> print cat
        4 Event(s) in Catalog:
        2011-01-07T01:29:49.760000Z | +49.520, +156.895 | 4.2 mb
        2011-01-07T01:19:16.660000Z | +20.123,  -45.656 | 5.5 MW
        2011-01-07T01:14:45.500000Z |  -3.268, +100.745 | 4.5 mb
        2011-01-07T01:14:01.280000Z | +36.095,  +27.550 | 4.0 mb

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
        :type eventid: str or int (dependent on data center), optional
        :param eventid: Select a specific event by ID; event identifiers are
            data center specific.
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
        :type filename: str or open file-like object
        :param filename: If given, the downloaded data will be saved there
            instead of being parse to an ObsPy object. Thus it will contain the
            raw data from the webservices.


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
            cat = obspy.readEvents(data_stream, format="quakeml")
            data_stream.close()
            return cat

    def get_stations(self, starttime=None, endtime=None, startbefore=None,
                     startafter=None, endbefore=None, endafter=None,
                     network=None, station=None, location=None, channel=None,
                     minlatitude=None, maxlatitude=None, minlongitude=None,
                     maxlongitude=None, latitude=None, longitude=None,
                     minradius=None, maxradius=None, level=None,
                     includerestricted=None, includeavailability=None,
                     updatedafter=None, filename=None, **kwargs):
        """
        Query the station service of the client.

        >>> client = Client("IRIS")
        >>> stationxml_string = client.get_stations(
        ...     latitude=-56.1, longitude=-26.7, maxradius=15)
        >>> for line in stationxml_string.splitlines()[:6]:
        ...     print line  # doctest: +ELLIPSIS
        <?xml version="1.0" encoding="ISO-8859-1"?>
        <BLANKLINE>
        <FDSNStationXML ...>
         <Source>IRIS-DMC</Source>
         <Sender>IRIS-DMC</Sender>
         <Module>IRIS WEB SERVICE: fdsnws-station | version: 1.0.7</Module>
        >>> stationxml_string = client.get_stations(
        ...     starttime=UTCDateTime("2013-01-01"), network="IU",
        ...     sta="ANMO", level="channel")
        >>> for line in stationxml_string.splitlines()[:6]:
        ...     print line  # doctest: +ELLIPSIS
        <?xml version="1.0" encoding="ISO-8859-1"?>
        <BLANKLINE>
        <FDSNStationXML ...>
         <Source>IRIS-DMC</Source>
         <Sender>IRIS-DMC</Sender>
         <Module>IRIS WEB SERVICE: fdsnws-station | version: 1.0.7</Module>

        :type starttime:
        :param starttime: Limit to metadata epochs starting on or after the
            specified start time.
        :type endtime:
        :param endtime: Limit to metadata epochs ending on or before the
            specified end time.
        :type startbefore:
        :param startbefore: Limit to metadata epochs starting before specified
            time.
        :type startafter:
        :param startafter: Limit to metadata epochs starting after specified
            time.
        :type endbefore:
        :param endbefore: Limit to metadata epochs ending before specified
            time.
        :type endafter:
        :param endafter: Limit to metadata epochs ending after specified time.
        :type network:
        :param network: Select one or more network codes. Can be SEED network
            codes or data center defined codes. Multiple codes are
            comma-separated.
        :type station:
        :param station: Select one or more SEED station codes. Multiple codes
            are comma-separated.
        :type location:
        :param location: Select one or more SEED location identifiers. Multiple
            identifiers are comma-separated. As a special case “--“ (two
            dashes) will be translated to a string of two space characters to
            match blank location IDs.
        :type channel:
        :param channel: Select one or more SEED channel codes. Multiple codes
            are comma-separated.
        :type minlatitude:
        :param minlatitude: Limit to stations with a latitude larger than the
            specified minimum.
        :type maxlatitude:
        :param maxlatitude: Limit to stations with a latitude smaller than the
            specified maximum.
        :type minlongitude:
        :param minlongitude: Limit to stations with a longitude larger than the
            specified minimum.
        :type maxlongitude:
        :param maxlongitude: Limit to stations with a longitude smaller than
            the specified maximum.
        :type latitude:
        :param latitude: Specify the latitude to be used for a radius search.
        :type longitude:
        :param longitude: Specify the longitude to the used for a radius
            search.
        :type minradius:
        :param minradius: Limit results to stations within the specified
            minimum number of degrees from the geographic point defined by the
                    latitude and longitude parameters.
        :type maxradius:
        :param maxradius: Limit results to stations within the specified
            maximum number of degrees from the geographic point defined by the
            latitude and longitude parameters.
        :type level:
        :param level: Specify the level of detail for the results.
        :type includerestricted:
        :param includerestricted: Specify if results should include information
            for restricted stations.
        :type includeavailability:
        :param includeavailability: Specify if results should include
            information about time series data availability.
        :type updatedafter:
        :param updatedafter: Limit to metadata updated after specified date;
            updates are data center specific.
        :type filename: str or open file-like object
        :param filename: If given, the downloaded data will be saved there
            instead of being parse to an ObsPy object. Thus it will contain the
            raw data from the webservices.

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

        # Special location handling. Convert empty strings to "--".
        if "location" in kwargs and not kwargs["location"]:
            kwargs["location"] = "--"

        url = self._create_url_from_parameters(
            "station", DEFAULT_PARAMETERS['station'], kwargs)

        data_stream = self._download(url)
        data_stream.seek(0, 0)
        if filename:
            self._write_to_file_object(filename, data_stream)
            data_stream.close()
        else:
            # XXX: Replace with StationXML reader once ready!
            station = data_stream.read()
            data_stream.close()
            return station

    def get_waveform(self, network, station, location, channel, starttime,
                     endtime, quality=None, minimumlength=None,
                     longestonly=None, filename=None, **kwargs):
        """
        Query the dataselect service of the client.

        >>> client = Client("IRIS")
        >>> t1 = UTCDateTime("2010-02-27T06:30:00.000")
        >>> t2 = t1 + 1
        >>> st = client.get_waveform("IU", "ANMO", "00", "BHZ", t1, t2)
        >>> print st  # doctest: +ELLIPSIS
        1 Trace(s) in Stream:
        IU.ANMO.00.BHZ | 2010-02-27T06:30:00... | 20.0 Hz, 20 samples
        >>> st = client.get_waveform("IU", "ANMO", "00", "BH*", t1, t2)
        >>> print st  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        IU.ANMO.00.BH1 | 2010-02-27T06:30:00... | 20.0 Hz, 20 samples
        IU.ANMO.00.BH2 | 2010-02-27T06:30:00... | 20.0 Hz, 20 samples
        IU.ANMO.00.BHZ | 2010-02-27T06:30:00... | 20.0 Hz, 20 samples
        >>> st = client.get_waveform("IU", "A*", "*", "BHZ", t1, t2)
        >>> print st  # doctest: +ELLIPSIS
        7 Trace(s) in Stream:
        IU.ADK.00.BHZ  | 2010-02-27T06:30:00... | 20.0 Hz, 20 samples
        IU.ADK.10.BHZ  | 2010-02-27T06:30:00... | 40.0 Hz, 40 samples
        IU.AFI.00.BHZ  | 2010-02-27T06:30:00... | 20.0 Hz, 20 samples
        IU.AFI.10.BHZ  | 2010-02-27T06:30:00... | 40.0 Hz, 40 samples
        IU.ANMO.00.BHZ | 2010-02-27T06:30:00... | 20.0 Hz, 20 samples
        IU.ANMO.10.BHZ | 2010-02-27T06:30:00... | 40.0 Hz, 40 samples
        IU.ANTO.00.BHZ | 2010-02-27T06:30:00... | 20.0 Hz, 20 samples
        >>> st = client.get_waveform("IU", "A??", "?0", "BHZ", t1, t2)
        >>> print st  # doctest: +ELLIPSIS
        4 Trace(s) in Stream:
        IU.ADK.00.BHZ | 2010-02-27T06:30:00... | 20.0 Hz, 20 samples
        IU.ADK.10.BHZ | 2010-02-27T06:30:00... | 40.0 Hz, 40 samples
        IU.AFI.00.BHZ | 2010-02-27T06:30:00... | 20.0 Hz, 20 samples
        IU.AFI.10.BHZ | 2010-02-27T06:30:00... | 40.0 Hz, 40 samples

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
        :type quality: str, optional
        :param quality: Select a specific SEED quality indicator, handling is
            data center dependent.
        :type minimumlength: float, optional
        :param minimumlength: Limit results to continuous data segments of a
            minimum length specified in seconds.
        :type longestonly: bool, optional
        :param longestonly: Limit results to the longest continuous segment per
            channel.
        :type filename: str or open file-like object
        :param filename: If given, the downloaded data will be saved there
            instead of being parse to an ObsPy object. Thus it will contain the
            raw data from the webservices.

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

        data_stream = self._download(url)
        data_stream.seek(0, 0)
        if filename:
            self._write_to_file_object(filename, data_stream)
            data_stream.close()
        else:
            st = obspy.read(data_stream, format="MSEED")
            data_stream.close()
            return st

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
            key for key, value in service_params.iteritems()
            if value["required"] is True]
        for req_param in required_parameters:
            if req_param not in parameters:
                msg = "Parameter '%s' is required." % req_param
                raise TypeError(msg)

        final_parameter_set = {}

        # Now loop over all parameters, convert them and make sure they are
        # accepted by the service.
        for key, value in parameters.iteritems():
            if key not in service_params:
                # If it is not in the service but in the default parameters
                # raise a warning.
                if key in default_params:
                    msg = ("The standard parameter '%s' is not supporte by "
                           "the webservice. It will be silently ignored." %
                           key)
                    warnings.warn(msg)
                    continue
                elif key in WADL_PARAMETERS_NOT_TO_BE_PARSED:
                    msg = ("The parameter '%s' is ignored because it is not "
                           "useful within ObsPy")
                    warnings.warn(msg)
                    continue
                # Otherwise raise an error.
                else:
                    msg = \
                        "The parameter '%s' is not supported by the service." \
                        % key
                    raise TypeError(msg)
            # Now attempt to convert the parameter to the correct type.
            this_type = service_params[key]["type"]
            try:
                value = this_type(value)
            except:
                msg = "'%s' could not be converted to type '%s'." % (
                    str(value), this_type.__name__)
                raise TypeError(msg)
            # Now convert to a string that is accepted by the webservice.
            value = convert_to_string(value)
            if isinstance(value, basestring):
                if not value:
                    continue
            final_parameter_set[key] = value

        return self._build_url(service, "query",
                               parameters=final_parameter_set)

    def __str__(self):
        ret = ("FDSN Webservice Client (base url: {url})\n"
               "Available Services: '{services}'\n\n"
               "Use e.g. client.help('dataselect') for the\n"
               "parameter description of the individual services\n"
               "or client.help() for parameter description of\n"
               "all webservices.".format(
                   url=self.base_url,
                   services="', '".join(self.services.keys())))
        return ret

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
            services = self.services.keys()
        elif service in DEFAULT_PARAMETERS:
            services = [service]
        else:
            raise NotImplementedError

        for service in services:
            SERVICE_DEFAULT = DEFAULT_PARAMETERS[service]

            print "Parameter description for the '%s' service of '%s':" % (
                service, self.base_url)

            # Loop over all parameters and group them in three list: available
            # default parameters, missing default parameters and additional
            # parameters
            available_default_parameters = []
            missing_default_parameters = []
            additional_parameters = []

            printed_something = False

            for name in SERVICE_DEFAULT:
                if name in self.services[service]:
                    available_default_parameters.append(name)
                else:
                    missing_default_parameters.append(name)

            for name in self.services[service].iterkeys():
                if name not in SERVICE_DEFAULT:
                    additional_parameters.append(name)

            def _print_param(name):
                param = self.services[service][name]
                name = "%s (%s)" % (name, param["type"].__name__)
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
                    doc_title = wrap_long_string(param["doc_title"],
                                                 prefix="        ")
                    doc_title = "\n" + doc_title
                else:
                    doc_title = ""

                print "    {name}{req_def}{doc_title}".format(
                    name=name, req_def=req_def, doc_title=doc_title)

            if additional_parameters:
                printed_something = True
                print ("The service offers the following "
                       "non-standard parameters:")
                for name in additional_parameters:
                    _print_param(name)

            if missing_default_parameters:
                printed_something = True
                print("WARNING: The service does not offer the following "
                      "standard parameters: %s" %
                      ", ".join(missing_default_parameters))

            if service == "event" and \
                    "available_event_catalogs" in self.services:
                printed_something = True
                print("Available catalogs: %s" %
                      ", ".join(
                          self.services["available_event_catalogs"]))

            if service == "event" and \
                    "available_event_contributors" in self.services:
                printed_something = True
                print("Available catalogs: %s" %
                      ", ".join(
                          self.services["available_event_contributors"]))

            if printed_something is False:
                print("No derivations from standard detected")

    def _download(self, url):
        code, data = download_url(url, headers=self.request_headers,
                                  debug=self.debug, return_string=False)
        # No data.
        if code == 204:
            raise FDSNException("No data available for request.")
        elif code == 400:
            msg = "Bad request. Please contact the developers."
            raise NotImplementedError(msg)
        elif code == 401:
            raise FDSNException("Unauthorized, authentication required.")
        elif code == 403:
            raise FDSNException("Authentication failed.")
        elif code == 413:
            raise FDSNException("Request would result in too much data. "
                                "Denied by the datacenter. Split the request "
                                "in smaller parts")
        # Request URI too large.
        elif code == 414:
            msg = ("The request URI is too large. Please contact the ObsPy "
                   "developers.")
            raise NotImplementedError(msg)
        elif code == 500:
            raise FDSNException("Service responds: Internal server error")
        elif code == 503:
            raise FDSNException("Service temporarily unavailable")

        return data

    def _build_url(self, resource_type, service, parameters={}):
        """
        Builds a correct URL.
        """
        return build_url(self.base_url, self.major_version, resource_type,
                         service, parameters)

    def _discover_services(self):
        """
        Automatically discovers available services.

        They are discovered by downloading the corresponding WADL files. If a
        WADL does not exist, the services are assumed to be non-existent.
        """
        dataselect_url = self._build_url("dataselect", "application.wadl")
        station_url = self._build_url("station", "application.wadl")
        event_url = self._build_url("event", "application.wadl")
        catalog_url = self._build_url("event", "catalogs")
        contributor_url = self._build_url("event", "contributors")
        urls = (dataselect_url, station_url, event_url, catalog_url,
                contributor_url)

        # Request all in parallel.
        wadl_queue = Queue.Queue()

        headers = self.request_headers
        debug = self.debug

        def get_download_thread(url):
            class ThreadURL(threading.Thread):
                def run(self):
                    code, data = download_url(url, headers=headers,
                                              debug=debug)
                    if code == 200:
                        wadl_queue.put((url, data))
                    else:
                        wadl_queue.put((url, None))
            return ThreadURL()

        threads = map(get_download_thread, urls)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(15)

        self.services = {}
        for _ in range(wadl_queue.qsize()):
            item = wadl_queue.get()
            url, wadl = item
            if wadl is None:
                continue
            if "dataselect" in url:
                self.services["dataselect"] = WADLParser(wadl).parameters
                if self.debug is True:
                    print "Discovered dataselect service"
            elif "event" in url and "application.wadl" in url:
                self.services["event"] = WADLParser(wadl).parameters
                if self.debug is True:
                    print "Discovered event service"
            elif "station" in url:
                self.services["station"] = WADLParser(wadl).parameters
                if self.debug is True:
                    print "Discovered station service"
            elif "event" in url and "catalogs" in url:
                try:
                    self.services["available_event_catalogs"] = \
                        parse_simple_xml(wadl)["catalogs"]
                except ValueError:
                    msg = "Could not parse the catalogs at '%s'."
                    warnings.warn(msg)

            elif "event" in url and "contributors" in url:
                try:
                    self.services["available_event_contributors"] = \
                        parse_simple_xml(wadl)["contributors"]
                except ValueError:
                    msg = "Could not parse the contributors at '%s'."
                    warnings.warn(msg)
        if not self.services:
            msg = ("No FDSN services could be discoverd at '%s'. This could "
                   "be due to a temporary service outage or an invalid FDSN "
                   "service address." % self.base_url)
            raise FDSNException(msg)


def convert_to_string(value):
    """
    Takes any value and converts it to a string compliant with the FDSN
    webservices.

    Will raise a ValueError if the value could not be converted.

    >>> convert_to_string("abcd")
    'abcd'
    >>> convert_to_string(1)
    '1'
    >>> convert_to_string(1.2)
    '1.2'
    >>> convert_to_string(obspy.UTCDateTime(2012, 1, 2, 3, 4, 5, 666666))
    '2012-01-02T03:04:05.666666'
    >>> convert_to_string(True)
    'true'
    >>> convert_to_string(False)
    'false'
    """
    if isinstance(value, basestring):
        return value
    # Boolean test must come before integer check!
    elif isinstance(value, bool):
        return str(value).lower()
    elif isinstance(value, int):
        return str(value)
    elif isinstance(value, float):
        return str(value)
    elif isinstance(value, obspy.UTCDateTime):
        return str(value).replace("Z", "")


def build_url(base_url, major_version, resource_type, service, parameters={}):
    """
    URL builder for the FDSN webservices.

    Built as a separate function to enhance testability.

    >>> build_url("http://service.iris.edu", 1, "dataselect", \
                  "application.wadl")
    'http://service.iris.edu/fdsnws/dataselect/1/application.wadl'

    >>> build_url("http://service.iris.edu", 1, "dataselect", \
                  "query", {"cha": "EHE"})
    'http://service.iris.edu/fdsnws/dataselect/1/query?cha=EHE'
    """
    # Only allow certain resource types.
    if resource_type not in ["dataselect", "event", "station"]:
        msg = "Resource type '%s' not allowed. Allowed resource types: \n%s" %\
            (resource_type, ",".join(("dataselect", "event", "station")))
        raise ValueError(msg)

    url = "/".join((base_url, "fdsnws", resource_type,
                    str(major_version), service))
    if parameters:
        url = "?".join((url, urllib.urlencode(parameters)))
    return url


def download_url(url, timeout=10, headers={}, debug=False, return_string=True):
    """
    Returns a pair of tuples.

    The first one is the returned HTTP code and the second the data as
    string.

    Will return a touple of Nones if the service could not be found.
    """
    if debug is True:
        print "Downloading %s" % url

    try:
        url_obj = urllib2.urlopen(urllib2.Request(url=url, headers=headers),
                                  timeout=timeout)
    # Catch HTTP errors.
    except urllib2.HTTPError as e:
        if debug is True:
            print("HTTP error %i while downloading '%s': %s" %
                  (e.code, url, e.read()))
        return e.code, None
    except Exception as e:
        if debug is True:
            print "Error while downloading: %s" % url
        return None, None

    code = url_obj.getcode()
    if return_string is False:
        data = BytesIO(url_obj.read())
    else:
        data = url_obj.read()

    if debug is True:
        print "Downloaded %s with HTTP code: %i" % (url, code)

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
                raise FDSNException(msg)
    # short aliases are not mentioned in the downloaded WADLs, so we have
    # to map it here according to the official FDSN WS documentation
    for key in list(kwargs.keys()):
        if key in PARAMETER_ALIASES:
            value = kwargs.pop(key)
            if value is not None:
                kwargs[PARAMETER_ALIASES[key]] = value

    for param in DEFAULT_PARAMETERS[service]:
        param = PARAMETER_ALIASES.get(param, param)
        value = locs[param]
        if value is not None:
            kwargs[param] = value


def parse_simple_xml(xml_string):
    """
    Simple helper function for parsing the Catalog and Contributor availability
    files.

    Parses XMLs of the form

    <Bs>
        <total>4</total>
        <B>1</B>
        <B>2</B>
        <B>3</B>
        <B>4</B>
    <Bs>

    and return a dictionary with a single item:

    {"Bs": set(("1", "2", "3", "4"))}
    """
    root = etree.fromstring(xml_string.strip())

    if not root.tag.endswith("s"):
        msg = "Could not parse the XML."
        raise ValueError(msg)
    child_tag = root.tag[:-1]
    children = [i.text for i in root if i.tag == child_tag]

    return {root.tag.lower(): set(children)}


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
